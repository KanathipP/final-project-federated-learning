from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
from typing import Dict, Any, List, Optional
import os
import re

# -------- Settings --------
NAMESPACE = os.getenv("TARGET_NAMESPACE", "flwr")
LABEL_SELECTOR = os.getenv(
    "LABEL_SELECTOR",
    "app.kubernetes.io/component in (clientapp,serverapp)"
)
CORS_ALLOW_ORIGINS = [o.strip() for o in os.getenv(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173"
).split(",") if o.strip()]

app = FastAPI(title="log-gateway", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS if CORS_ALLOW_ORIGINS else ["*"],
    allow_credentials=False,          # ไม่ใช้ cookie ในเบราว์เซอร์
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Startup --------
@app.on_event("startup")
def _startup():
    try:
        config.load_incluster_config()
    except Exception:
        config.load_kube_config()

# -------- Helpers: parse resources --------
_CPU_RE = re.compile(r"^(\d+)(m?)$")
_MEM_RE = re.compile(r"^(\d+)(Ei|Pi|Ti|Gi|Mi|Ki)?$")
_MEM_FACTORS = {"Ki":1024,"Mi":1024**2,"Gi":1024**3,"Ti":1024**4,"Pi":1024**5,"Ei":1024**6}

def parse_cpu_to_mcores(val: Optional[str]) -> Optional[int]:
    """
    แปลงปริมาณ CPU ของ K8s เป็น millicores (m)
    รองรับหน่วย: n (nano), u/µ (micro), m (milli), หรือไม่มีหน่วย (= cores)
    ตัวอย่าง:
      "250m" -> 250
      "2"    -> 2000
      "142536626n" -> 142   (≈ 142.536626 m, ปัดลง)
      "750000u" -> 750
    """
    if val is None or val == "":
        return None

    s = str(val).strip()
    # เคสจำนวนเต็มล้วน (ถือเป็น cores)
    if s.isdigit():
        return int(s) * 1000

    # หน่วย nano
    if s.endswith("n"):
        n = int(s[:-1]) if s[:-1] else 0
        return n // 1_000_000  # 1e9 n = 1000 m → /1e6

    # หน่วย micro (ทั้ง 'u' และสัญลักษณ์ 'µ')
    if s.endswith("u") or s.endswith("µ"):
        micro = int(s[:-1]) if s[:-1] else 0
        return micro // 1_000  # 1e6 u = 1000 m → /1e3

    # หน่วย milli
    if s.endswith("m"):
        # เผื่อมีทศนิยมเช่น "12.5m"
        num = s[:-1]
        return int(float(num))

    # ที่เหลือถือเป็น cores (อาจมีทศนิยม)
    return int(float(s) * 1000)

def parse_mem_to_bytes(val: Optional[str]) -> Optional[int]:
    if not val: return None
    m = _MEM_RE.match(val)
    if m:
        num, suf = m.groups()
        return int(num) * _MEM_FACTORS.get(suf, 1)
    for suf, f in _MEM_FACTORS.items():
        if val.endswith(suf): return int(float(val[:-len(suf)]) * f)
    return int(float(val))

def sum_none_safe(values: List[Optional[int]]) -> Optional[int]:
    vals = [v for v in values if v is not None]
    return sum(vals) if vals else None

def container_requests_limits(c: client.V1Container) -> Dict[str, Any]:
    res = c.resources or client.V1ResourceRequirements()
    req = res.requests or {}
    lim = res.limits or {}
    return {
        "requests": {"cpu_mcores": parse_cpu_to_mcores(req.get("cpu")), "memory_bytes": parse_mem_to_bytes(req.get("memory"))},
        "limits":   {"cpu_mcores": parse_cpu_to_mcores(lim.get("cpu")), "memory_bytes": parse_mem_to_bytes(lim.get("memory"))},
    }

# -------- Basic health --------
@app.get("/healthz")
def healthz():
    return {"ok": True}

# -------- Pods (list + inventory) --------
@app.get("/pods")
def list_pods(
    component: str = Query("", description="clientapp หรือ serverapp"),
    instance: str = Query("", description="เช่น '0','1' ตาม label app.kubernetes.io/instance"),
):
    v1 = client.CoreV1Api()
    parts = [LABEL_SELECTOR] if LABEL_SELECTOR else []
    if component: parts.append(f"app.kubernetes.io/component={component}")
    if instance:  parts.append(f"app.kubernetes.io/instance={instance}")
    selector = ",".join(parts) if parts else None

    pods = v1.list_namespaced_pod(namespace=NAMESPACE, label_selector=selector)
    items = []
    for p in pods.items:
        labels = p.metadata.labels or {}
        items.append({
            "name": p.metadata.name,
            "namespace": p.metadata.namespace,
            "nodeName": p.spec.node_name,
            "phase": p.status.phase,
            "labels": labels,
            "component": labels.get("app.kubernetes.io/component",""),
            "instance": labels.get("app.kubernetes.io/instance",""),
            "containers": [c.name for c in p.spec.containers],
        })
    return {"items": items}

@app.get("/inventory")
def inventory():
    v1 = client.CoreV1Api()
    pods = v1.list_namespaced_pod(namespace=NAMESPACE, label_selector=LABEL_SELECTOR)
    clients, servers = [], []
    for p in pods.items:
        labels = p.metadata.labels or {}
        item = {
            "name": p.metadata.name,
            "instance": labels.get("app.kubernetes.io/instance",""),
            "nodeName": p.spec.node_name,
            "phase": p.status.phase,
        }
        comp = labels.get("app.kubernetes.io/component","")
        (clients if comp == "clientapp" else servers if comp == "serverapp" else []).append(item)
    return {"clients": clients, "servers": servers}

# -------- Logs --------
@app.get("/pods/{pod}/log", response_class=PlainTextResponse)
def get_log(pod: str, container: str = "", tail: int = Query(200, ge=1, le=5000), previous: bool = False):
    v1 = client.CoreV1Api()
    try:
        return v1.read_namespaced_pod_log(
            name=pod, namespace=NAMESPACE, container=container or None,
            tail_lines=tail, previous=previous
        )
    except ApiException as e:
        # ส่งข้อความชัด ๆ กลับไปให้ FE
        msg = e.body or e.reason or str(e)
        raise HTTPException(status_code=e.status or 500, detail=msg)

# -------- Metrics: nodes --------
@app.get("/metrics/nodes")
def node_metrics():
    """
    ถ้ามี metrics-server: เติม usage_{cpu,memory}
    ถ้าไม่มี: คืน usage เป็น None แต่ยังคืน capacity/allocatable ได้
    """
    v1 = client.CoreV1Api()
    co = client.CustomObjectsApi()

    usage_by_node: Dict[str, Dict[str, Optional[int]]] = {}
    try:
        nm = co.list_cluster_custom_object("metrics.k8s.io","v1beta1","nodes")
        for i in nm.get("items", []):
            usage_by_node[i["metadata"]["name"]] = {
                "cpu_mcores": parse_cpu_to_mcores(i["usage"]["cpu"]),
                "memory_bytes": parse_mem_to_bytes(i["usage"]["memory"]),
            }
    except ApiException as e:
        if e.status not in (404, 503, 403):
            raise HTTPException(status_code=500, detail=f"metrics API error (nodes): ({e.status}) {e.reason}")
        # ไม่มี metrics-server / ยังไม่พร้อม → ปล่อย usage เป็น None

    nodes = v1.list_node()
    items = []
    for n in nodes.items:
        name = n.metadata.name
        cap_cpu  = parse_cpu_to_mcores((n.status.capacity or {}).get("cpu"))
        cap_mem  = parse_mem_to_bytes((n.status.capacity or {}).get("memory"))
        alloc_cpu = parse_cpu_to_mcores((n.status.allocatable or {}).get("cpu"))
        alloc_mem = parse_mem_to_bytes((n.status.allocatable or {}).get("memory"))
        used = usage_by_node.get(name, {})
        items.append({
            "name": name,
            "cpu": {"capacity_mcores": cap_cpu, "allocatable_mcores": alloc_cpu, "usage_mcores": used.get("cpu_mcores")},
            "memory": {"capacity_bytes": cap_mem, "allocatable_bytes": alloc_mem, "usage_bytes": used.get("memory_bytes")},
            "labels": n.metadata.labels or {},
        })
    return {"items": items}

# -------- Metrics: pods --------
@app.get("/metrics/pods")
def pod_metrics(namespace: str = NAMESPACE):
    """
    ถ้ามี metrics-server: เติม usage ต่อ container/รวมทั้ง pod
    ถ้าไม่มี: usage เป็น 0/None แต่ยังคืนข้อมูล requests/limits ได้
    """
    v1 = client.CoreV1Api()
    co = client.CustomObjectsApi()

    usage_map: Dict[str, Dict[str, Dict[str, Optional[int]]]] = {}
    try:
        pm = co.list_namespaced_custom_object("metrics.k8s.io","v1beta1","pods", namespace)
        for item in pm.get("items", []):
            pod_name = item["metadata"]["name"]
            usage_map[pod_name] = {}
            for c in item.get("containers", []):
                usage_map[pod_name][c["name"]] = {
                    "cpu_mcores": parse_cpu_to_mcores(c["usage"]["cpu"]),
                    "memory_bytes": parse_mem_to_bytes(c["usage"]["memory"]),
                }
    except ApiException as e:
        if e.status not in (404, 503, 403):
            raise HTTPException(status_code=500, detail=f"metrics API error (pods): ({e.status}) {e.reason}")
        # ไม่มี metrics-server → ปล่อย usage_map ว่าง

    pods = v1.list_namespaced_pod(namespace=namespace, label_selector=LABEL_SELECTOR)
    results = []
    for p in pods.items:
        labels = p.metadata.labels or {}
        conts = []
        req_cpu, req_mem, lim_cpu, lim_mem = [], [], [], []
        for c in p.spec.containers:
            rl = container_requests_limits(c)
            cont_usage = usage_map.get(p.metadata.name, {}).get(c.name, {}) or {}
            conts.append({
                "name": c.name,
                "usage": cont_usage or None,
                "requests": rl["requests"],
                "limits": rl["limits"],
            })
            req_cpu.append(rl["requests"]["cpu_mcores"])
            req_mem.append(rl["requests"]["memory_bytes"])
            lim_cpu.append(rl["limits"]["cpu_mcores"])
            lim_mem.append(rl["limits"]["memory_bytes"])

        results.append({
            "name": p.metadata.name,
            "namespace": p.metadata.namespace,
            "nodeName": p.spec.node_name,
            "phase": p.status.phase,
            "labels": labels,
            "component": labels.get("app.kubernetes.io/component",""),
            "instance": labels.get("app.kubernetes.io/instance",""),
            "containers": conts,
            "requests_total": {"cpu_mcores": sum_none_safe(req_cpu), "memory_bytes": sum_none_safe(req_mem)},
            "limits_total":   {"cpu_mcores": sum_none_safe(lim_cpu), "memory_bytes": sum_none_safe(lim_mem)},
            "usage_total": {
                "cpu_mcores": sum((x.get("usage") or {}).get("cpu_mcores", 0) for x in conts),
                "memory_bytes": sum((x.get("usage") or {}).get("memory_bytes", 0) for x in conts),
            },
        })
    return {"items": results}
