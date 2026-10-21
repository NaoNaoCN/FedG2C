"""Generate native draw.io sequence diagrams from the current source locations.

Run with Python; only the standard library is used. Source files are read, never
imported or changed. Symbol locations are resolved by AST to avoid stale labels.
"""
from __future__ import annotations

import ast
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "docs" / "fedg2c-stage-sequences.drawio"
SYMBOLS = {}
REFS = []
for source in list((REPO / "fedg2c").rglob("*.py")) + [
    REPO / name for name in (
        "pretrain_fedg2c.py", "pretrain_federated_fedg2c.py",
        "aggregate_fedg2c.py", "finetune_fedg2c.py", "model/graph_encoders.py",
    )
]:
    path = source.relative_to(REPO).as_posix()
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            SYMBOLS[path, node.name] = node.lineno
        if isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    SYMBOLS[path, f"{node.name}.{method.name}"] = method.lineno


def source(path, symbol):
    line = SYMBOLS[path, symbol]
    ref = (path, symbol, line)
    if ref not in REFS:
        REFS.append(ref)
    return ref


def esc(text):
    return html.escape(str(text), quote=True)


def rich(title, detail="", ref=None):
    result = f"<b>{esc(title)}</b>"
    if detail:
        result += "<br>" + esc(detail)
    if ref:
        result += f'<br><span style="font-size:12px;color:#64748B">{esc(ref[0])}:{ref[2]}</span>'
    return result


BASE = "html=1;whiteSpace=wrap;fontFamily=Microsoft YaHei;fontColor=#172B4D;"
COLORS = {
    "blue": ("#EAF2FF", "#6A8FCC"),
    "green": ("#E9F5ED", "#68A27C"),
    "yellow": ("#FFF4D8", "#C6A153"),
    "purple": ("#F0EBFA", "#9B84C4"),
    "gray": ("#F3F5F8", "#9AA8BA"),
}


class Sequence:
    def __init__(self, mxfile, key, title, subtitle, contract):
        self.key, self.counter = key, 0
        self.width = 3260
        self.page = ET.SubElement(mxfile, "diagram", id=key, name=title)
        self.model = ET.SubElement(self.page, "mxGraphModel", {
            "dx": "1600", "dy": "1000", "grid": "1", "gridSize": "10",
            "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1",
            "fold": "1", "page": "0", "pageScale": "1", "pageWidth": str(self.width),
            "pageHeight": "6000", "background": "#FFFFFF", "math": "0",
            "shadow": "0", "adaptiveColors": "auto",
        })
        self.root = ET.SubElement(self.model, "root")
        ET.SubElement(self.root, "mxCell", id="0")
        ET.SubElement(self.root, "mxCell", id="1", parent="0")
        self.text(60, 28, 3140, 58, rich(title), 30, bold=True)
        self.text(60, 91, 3140, 45, esc(subtitle), 17)
        self.box(60, 147, 3140, 77, rich("阶段边界", contract), "blue", 15)
        self.text(65, 238, 3130, 43,
                  "实线箭头＝同步调用　　虚线箭头＝返回　　折返箭头＝对象内部执行 / 调用标注的工具函数　　"
                  "loop / alt / opt＝循环 / 互斥分支 / 可选步骤；同页下方 ref 为被调用过程的展开。", 14)
        self.y = 310
        self.lanes = {}
        self.lifeline_edges = []
        self.messages = 0
        self.message_records = []
        self.section_ends = []

    def cell(self, value, style, x, y, w, h, ref=None, ident=None):
        self.counter += 1
        ident = ident or f"{self.key}-{self.counter}"
        attrs = {"id": ident, "parent": "1", "value": value, "style": BASE + style, "vertex": "1"}
        if ref:
            attrs["tooltip"] = f"{ref[0]}:{ref[2]} | {ref[1]}"
        node = ET.SubElement(self.root, "mxCell", attrs)
        ET.SubElement(node, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"})
        return node

    def text(self, x, y, w, h, value, size=14, bold=False, ref=None, align="left"):
        return self.cell(value, f"text;strokeColor=none;fillColor=none;align={align};verticalAlign=middle;"
                         f"fontSize={size};spacing=3;fontStyle={1 if bold else 0};", x, y, w, h, ref)

    def box(self, x, y, w, h, value, color="gray", size=14, ref=None):
        fill, stroke = COLORS[color]
        return self.cell(value, f"rounded=1;arcSize=10;fillColor={fill};strokeColor={stroke};strokeWidth=1.3;"
                         f"align=left;verticalAlign=middle;spacing=12;fontSize={size};", x, y, w, h, ref)

    def line(self, x1, y1, x2, y2, dashed=False, arrow=False, points=None, color="#52657F"):
        self.counter += 1
        style = BASE + f"strokeColor={color};strokeWidth={1.5 if arrow else 1};"
        style += f"endArrow={'open' if dashed and arrow else 'block' if arrow else 'none'};endFill={0 if dashed else 1};"
        style += "labelBackgroundColor=default;defaultLabelBackgroundColor=gap;"
        if dashed:
            style += "dashed=1;dashPattern=6 5;"
        edge = ET.SubElement(self.root, "mxCell", id=f"{self.key}-{self.counter}", parent="1", style=style, edge="1")
        geo = ET.SubElement(edge, "mxGeometry", relative="1", **{"as": "geometry"})
        ET.SubElement(geo, "mxPoint", x=str(x1), y=str(y1), **{"as": "sourcePoint"})
        ET.SubElement(geo, "mxPoint", x=str(x2), y=str(y2), **{"as": "targetPoint"})
        if points:
            array = ET.SubElement(geo, "Array", **{"as": "points"})
            for px, py in points:
                ET.SubElement(array, "mxPoint", x=str(px), y=str(py))
        return edge

    def section(self, title, participants, description=""):
        self.finish_lanes()
        self.box(60, self.y, 3140, 55, rich(title), "gray", 20)
        self.y += 67
        if description:
            self.text(65, self.y, 3130, 48, esc(description), 14)
            self.y += 62
        top = self.y
        count = len(participants)
        # Wide gaps keep complete method names and source references legible.
        centers = [220 + i * 2800 / (count - 1) for i in range(count)] if count > 1 else [1630]
        self.lanes = {}
        for x, (key, name, detail, color, ref) in zip(centers, participants):
            self.lanes[key] = x
            self.box(x - 165, top, 330, 102, rich(name, detail, ref), color, 15, ref)
            edge = self.line(x, top + 103, x, top + 500, dashed=True, color="#A0AABC")
            self.lifeline_edges.append(edge)
        self.y = top + 184

    def finish_lanes(self):
        if not self.lifeline_edges:
            return
        for edge in self.lifeline_edges:
            edge.find("mxGeometry/mxPoint[@as='targetPoint']").set("y", str(self.y + 8))
        self.lifeline_edges.clear()
        self.section_ends.append(self.y)
        self.y += 55

    def message(self, src, dst, title, detail="", ref=None, code=None, ret=False, gap=None):
        self.messages += 1
        number = code or f"{self.key.upper()}{self.messages:02d}"
        value = rich(f"{number}  {title}", detail, ref)
        x1, x2, y = self.lanes[src], self.lanes[dst], self.y
        color = "#73839A" if ret else "#385A80"
        if src == dst:
            self.line(x1, y, x1, y + 24, arrow=True,
                      points=[(x1 + 34, y), (x1 + 34, y + 24)], color=color)
            # Self-call labels are placed to the right, except at the right edge.
            label_x = x1 + 49 if x1 < 2780 else x1 - 364
            self.text(label_x, y - 61, 315, 81, value, 14, ref=ref)
        else:
            self.line(x1, y, x2, y, dashed=ret, arrow=True, color=color)
            # Small activation slivers show which object receives control.
            if not ret:
                self.cell("", "fillColor=#FFFFFF;strokeColor=#6A8FCC;strokeWidth=1;", x2-5, y-4, 10, 30)
            width = min(abs(x2-x1) - 36, 1120)
            self.text((x1+x2-width)/2, y - 70, width, 68, value, 14, ref=ref, align="center")
        self.message_records.append({"number":number,"title":title,"detail":detail,"source":ref,"y":y})
        self.y += gap if gap is not None else (84 if ret else 108)

    def note(self, title, text, color="yellow", height=91, left=70, width=3120):
        self.box(left, self.y-28, width, height, rich(title, text), color, 14)
        self.y += height + 60

    def fragment(self, kind, condition, left=60, right=3200):
        top = self.y - 28
        rect = self.cell("", "fillColor=none;strokeColor=#A7B4C5;strokeWidth=1.2;rounded=0;", left, top, right-left, 200)
        self.box(left+1, top+1, min(1480, right-left-2), 44, rich(f"{kind}  [{condition}]"), "gray", 14)
        self.y += 105
        return rect, top

    def alternative(self, condition, left=60, right=3200):
        self.line(left, self.y-24, right, self.y-24, dashed=True, color="#A7B4C5")
        self.box(left+1, self.y-23, min(1480,right-left-2), 43, rich(f"else  [{condition}]"), "gray", 14)
        self.y += 112

    def end(self, frame):
        rect, top = frame
        rect.find("mxGeometry").set("height", str(self.y-top-15))
        self.y += 40

    def finish(self, note):
        self.finish_lanes()
        self.box(60, self.y, 3140, 102, rich("代码对照与阅读提示", note), "green", 15)
        self.y += 145
        self.model.set("pageHeight", str(self.y))
        self.model.set("dy", "1000")


def r(file, symbol):
    return source("fedg2c/" + file + ".py", symbol)


def make_a(mx):
    s = Sequence(mx, "a", "阶段 A｜客户端本地预训练 · 代码时序图",
                 "当前 edge_prediction 路径；默认 GraphSAGE 预测器。上半页读训练顺序，下半页展开共享专家与组合器。",
                 "输入：下发共享参数 θ、客户端私有图；输出：本轮 experts.* + composer.*。私有适配器、预测器及动量副本留在客户端。")
    s.section("A-1｜客户端生命周期与一次本地训练步", [
        ("entry", "训练入口", "联邦 main / 独立 edge_entry.run", "gray", source("pretrain_federated_fedg2c.py","main")),
        ("client", "EdgePredictionClient", "下发 → 预训练 → 导出", "blue", r("edge_client","EdgePredictionClient")),
        ("sampler", "NeighborhoodEdgeSampler", "私有图 → 双视图 + 查询索引", "green", r("edge_views","NeighborhoodEdgeSampler")),
        ("model", "NodewisePredictionModel", "组织预测、参考与损失", "yellow", r("edge_model","NodewisePredictionModel")),
        ("main", "main_branch", "MaskedEncodingBranch + 私有适配器", "yellow", r("edge_model","MaskedEncodingBranch")),
        ("shared", "SharedBackbone × 2", "主分支 / 动量分支各自持有实例", "purple", r("model","SharedBackbone")),
        ("pred", "预测器", "GraphSAGEPredictor / Transformer 对照", "yellow", r("graph_predictor","GraphSAGEPredictor")),
        ("ema", "momentum_branch", "MaskedEncodingBranch；无梯度参考", "blue", r("edge_model","NodewisePredictionModel.update_momentum")),
    ], f"入口：pretrain_federated_fedg2c.py:{source('pretrain_federated_fedg2c.py','main')[2]}；独立入口：pretrain_fedg2c.py:{source('pretrain_fedg2c.py','main')[2]} → fedg2c/edge_entry.py:{r('edge_entry','run')[2]}。这里完整展开一个客户端。")
    s.message("entry","client","EdgePredictionClient(...)","构造客户端；deepcopy(shared)",r("edge_client","EdgePredictionClient.__init__"),code="A01")
    s.message("client","model","NodewisePredictionModel(...)","建立主分支、动量副本和私有预测器",r("edge_model","NodewisePredictionModel.__init__"),code="A02")
    s.note("初始化细节", "main_branch = MaskedEncodingBranch(shared, schema)；momentum_branch = deepcopy(main_branch).requires_grad_(False).eval()；默认 predictor_type='graphsage'。",height=78)
    s.message("entry","client","set_pretrain_model(state)","加载本轮共同的共享模型",r("edge_client","EdgePredictionClient.set_pretrain_model"),code="A03")
    s.message("client","shared","main_branch.shared.load_state_dict(...) ","仅覆盖共享参数；私有模块继续保留",r("edge_client","EdgePredictionClient.set_pretrain_model"),code="A04")
    s.message("client","model","reset_momentum()", "同步整条主分支到动量分支",r("edge_model","NodewisePredictionModel.reset_momentum"),code="A05")
    s.message("model","ema","load_state_dict(main_branch.state_dict())", "包括主分支共享权重与私有输入适配器",r("edge_model","NodewisePredictionModel.reset_momentum"),code="A06")
    s.message("entry","client","pretrain(graphs_factory, ...)","每次调用新建 AdamW；model.train()",r("edge_client","EdgePredictionClient.pretrain"),code="A07")
    loop=s.fragment("loop","epoch → graph；每张图建立一次 sampler",left=430)
    s.message("client","sampler","NeighborhoodEdgeSampler(graph)","每张 graph 建立一次邻接采样器",r("context","ContextSampler.__init__"),code="A08")
    views=s.fragment("loop","views_per_graph；max_steps 可提前结束 pretrain",left=460)
    s.message("client","sampler","sample_edge_view(config, channels, rng)","每个 view 重新采样",r("edge_views","NeighborhoodEdgeSampler.sample_edge_view"),code="A09")
    s.message("sampler","sampler","neighborhood() → induced_view()", "取根节点的有界局部参考图",r("edge_views","NeighborhoodEdgeSampler.neighborhood"),code="A10")
    s.message("sampler","sampler","mask_edges() → neighborhood() → sample_feature_keep()", "先遮边，再在可见边上选 q，最后采样通道掩码",r("edge_views","mask_edges"),code="A11")
    s.message("sampler","client","EdgePredictionPair", "reference / visible 同节点顺序；q[Q]、feature_keep[C]",r("edge_views","EdgePredictionPair"),code="A12",ret=True)
    s.message("client","client","optimizer.zero_grad(set_to_none=True)","清除上一步梯度",r("edge_client","EdgePredictionClient.pretrain"),code="A13")
    s.message("client","model","forward(pair)","q 为可见图局部查询索引",r("edge_model","NodewisePredictionModel.forward"),code="A14")
    s.message("model","main","forward(visible_view, feature_keep)","只使用遮边后的可见图",r("edge_model","MaskedEncodingBranch.forward"),code="A15")
    s.message("main","main","FeatureAdapter.forward(raw)","连续：先遮原特征再投影；离散：先嵌入再遮通道",r("model","FeatureAdapter.forward"),code="A16")
    s.message("main","shared","ref A-S：SharedBackbone.forward(x, edges, edge)","私有 node / edge adapter 将特征对齐到 d 维",r("model","SharedBackbone.forward"),code="A17")
    s.message("shared","model","main_z [N,d]（经 main_branch 返回）", "按原节点顺序排列的可见图表示",r("edge_model","NodewisePredictionModel.forward"),code="A18",ret=True)
    s.message("model","model","random_walk_encoding(visible_edges, ...)","遮边后重算位置 P[N,L]；L=len(walk_steps)",r("structural_encoding","random_walk_encoding"),code="A19")
    branch=s.fragment("alt","predictor_type == 'graphsage'（默认）",left=1230)
    s.message("model","pred","GraphSAGEPredictor.forward(main_z, P, q, visible_edges)","嵌入 + 位置投影 → SAGEConv → ReLU → output → [q]",r("graph_predictor","GraphSAGEPredictor.forward"),code="A20")
    s.alternative("predictor_type == 'transformer'",left=1230)
    s.message("model","pred","NodeRepresentationPredictor.forward(main_z, P, q)","Transformer 对照分支",r("node_predictor","NodeRepresentationPredictor.forward"),code="A21")
    s.end(branch)
    s.message("pred","model","prediction [Q,d]", "与 q 的顺序一致",code="A22",ret=True)
    ng=s.fragment("group","torch.no_grad()：动量分支参考计算（每步都执行）",left=1230)
    s.message("model","ema","forward(reference_view)","完整局部参考边；不传 feature_keep",r("edge_model","MaskedEncodingBranch.forward"),code="A23")
    s.message("ema","shared","ref A-S：动量副本 SharedBackbone.forward(...) ","使用 EMA 参数；与主分支不是同一权重实例",r("model","SharedBackbone.forward"),code="A24")
    s.message("shared","ema","reference_z [N,d]",code="A25",ret=True)
    s.message("ema","model","reference_z [N,d]", "后续 layer_norm(reference_z)[q] → reference[Q,d]",r("edge_model","NodewisePredictionModel.forward"),code="A26",ret=True)
    s.end(ng)
    s.message("model","model","F.smooth_l1_loss(prediction, reference)","reference 已逐节点 LayerNorm 并索引 q；损失为标量",r("edge_model","NodewisePredictionModel.forward"),code="A27")
    s.message("model","client","loss, outputs", "outputs 内的 prediction / reference 已 detach",r("edge_model","NodewisePredictionModel.forward"),code="A28",ret=True)
    s.message("client","client","loss.backward() → clip_grad_norm_(..., 5) ","更新目标：主分支共享参数 + 私有适配器 + 预测器",r("edge_client","EdgePredictionClient.pretrain"),code="A29")
    s.message("client","client","optimizer.step()", "先完成主分支优化",r("edge_client","EdgePredictionClient.pretrain"),code="A30")
    s.message("client","model","update_momentum()", "必须在 optimizer.step() 之后",r("edge_model","NodewisePredictionModel.update_momentum"),code="A31")
    s.message("model","ema","EMA 参数；直接复制 buffers", "θ_ema ← m·θ_ema + (1−m)·θ_main",r("edge_model","NodewisePredictionModel.update_momentum"),code="A32")
    s.end(views)
    s.end(loop)
    s.message("client","entry","_summary(metrics, references)", "本地训练报告；停止条件达到则返回",r("edge_client","EdgePredictionClient._summary"),code="A33",ret=True)
    s.message("entry","client","get_pretrain_model(); get_local_message()", "前者保存磁盘；后者供联邦入口上传",r("edge_client","EdgePredictionClient.get_local_message"),code="A34")
    s.message("client","entry","共享 state / {num_samples:1, weight:[(name,tensor)]}","只导出 main_branch.shared → 进入阶段 B",r("messages","client_message"),code="A35",ret=True)
    s.note("外层顺序", "联邦入口 main：逐客户端执行 A → server.execute(...) 执行 B → 下一轮重新下发。独立 edge_entry.run 只保存各客户端 shared_backbone.pt / local_training_state.pt。", "green",height=84)

    s.section("ref A-S｜SharedBackbone.forward 的公共调用序列", [
        ("caller","调用分支","主分支或动量分支的私有 adapter 之后","gray",r("edge_model","MaskedEncodingBranch.forward")),
        ("shared","SharedBackbone","组织专家与关系组合","purple",r("model","SharedBackbone.forward")),
        ("expert","experts[i]","EdgePassingGraphEncoder","purple",r("model","EdgePassingGraphEncoder.encode")),
        ("rel","结构工具函数","structural_relations","green",r("context","structural_relations")),
        ("composer","RelationComposer","关系条件组合器","purple",r("composer","RelationComposer.forward")),
    ], "本段在 A17 和 A24 各调用一次。N=局部节点数，M=专家数，d=隐藏维度，H=头数，T=N·M；默认 d=64、M=4、H=4。")
    s.message("caller","shared","forward(x[N,d], edge_index[2,E], edge_attr?)",ref=r("model","SharedBackbone.forward"),code="AS01")
    s.message("shared","shared","encode_experts(x, edges, edge)",ref=r("model","SharedBackbone.encode_experts"),code="AS02")
    loop=s.fragment("loop","i = 0 … M−1；专家按 ModuleList 顺序执行",left=700)
    s.message("shared","expert","GraphEncoder.forward(...) → self.encode(...) ","通过继承分派到 EdgePassingGraphEncoder.encode",source("model/graph_encoders.py","GraphEncoder.forward"),code="AS03")
    s.message("expert","expert","逐层 layer → norm → 非末层 activation/dropout", "mysage 将 edge_attr 传给 EdgeAwareSAGEConv",r("model","EdgePassingGraphEncoder.encode"),code="AS04")
    s.message("expert","shared","专家节点表示 [N,d]",code="AS05",ret=True)
    s.end(loop)
    s.message("shared","shared","torch.stack(expert_outputs, dim=1)","得到 E[N,M,d]；每个节点有 M 个专家 token",r("model","SharedBackbone.encode_experts"),code="AS06")
    s.message("shared","rel","structural_relations(edges, N)","self / direct_edge / two_hop 三通道",r("context","structural_relations"),code="AS07")
    s.message("rel","shared","R[N,N,3]",code="AS08",ret=True)
    s.message("shared","composer","forward(E[N,M,d], R[N,N,3])", "展平 token[T,d]；Q/K/V 按头拆分",r("composer","RelationComposer.forward"),code="AS09")
    s.message("composer","composer","QKᵀ / √(d/H) + relation(R)","scores[H,T,T]；节点关系偏置扩展到专家 token 对",r("composer","RelationComposer.forward"),code="AS10")
    s.message("composer","composer","head_rms(scores, mask)","跨头 RMS 归一化得到 code[H,T,T]",r("composer","head_rms"),code="AS11")
    s.message("composer","composer","hypernetwork_contract(code, V)","两次收缩；默认中间 ReLU；out[T,H,d/H]",r("composer","hypernetwork_contract"),code="AS12")
    s.message("composer","composer","output → token 残差 → LayerNorm → 专家均值", "[T,d] → [N,M,d] → z[N,d]",r("composer","RelationComposer.forward"),code="AS13")
    s.message("composer","caller","z[N,d]（经 SharedBackbone 返回）",code="AS14",ret=True)
    s.note("组合器配置分支", "A-S 展开默认 hypernetwork_relu；hypernetwork_linear 省略中间 ReLU；standard_attention 使用 softmax + 单次加权；linear_gate 直接按节点给专家加权。准确分支见 RelationComposer.forward。",height=84)
    s.finish("从 A01 向下读；到 ref A-S 时可跳到下半页。源码标注为当前文件行号；图形提示包含完整类 / 方法名。UML 主线表达调用顺序，未画出每个隐式 Python 返回。")
    return s


def make_b(mx):
    s=Sequence(mx,"b","阶段 B｜服务器聚合 · 代码时序图",
               "按 response 模式展开；同时标出 fedavg 的提前返回和 output_only 的差异。探针评估调用在下半页展开。",
               "输入：所有客户端的共享参数 θ₁…θK；输出：θ*=Σwᵢθᵢ 与聚合报告。服务器只优化 logits[K]，客户端上传参数保持 detach。")
    s.section("B-1｜一次 server.execute(...) 的完整顺序",[
        ("entry","聚合入口","联邦 main / aggregate.main","gray",source("aggregate_fedg2c.py","main")),
        ("server","ResponsePreservingServer","聚合状态、缓存、损失和报告","blue",r("response_server","ResponsePreservingServer")),
        ("utils","消息 / 参数函数","解包、distribution、mix_parameters","purple",r("response_server","mix_parameters")),
        ("probe","探针生成函数","sample_probe_bank","green",r("server_probes","sample_probe_bank")),
        ("eval","ProbeEvaluator","通过 functional_call 评估候选参数","purple",r("server_probes","ProbeEvaluator")),
        ("optim","torch.optim.Adam","仅持有 logits[K]","yellow",r("response_server","ResponsePreservingServer.execute")),
    ],f"联邦入口：pretrain_federated_fedg2c.py:{source('pretrain_federated_fedg2c.py','main')[2]}；独立入口：aggregate_fedg2c.py:{source('aggregate_fedg2c.py','main')[2]} → server_io.load_uploads（fedg2c/server_io.py:{r('server_io','load_uploads')[2]}）。")
    s.message("entry","server","execute(uploads, round_id)","联邦 messages 映射；或独立入口的 state 列表",r("response_server","ResponsePreservingServer.execute"),code="B01")
    opt=s.fragment("opt","uploads 是 Mapping（联邦运行消息）",left=560)
    s.message("server","utils","states_from_client_messages(uploads)","校验 num_samples=1 与 weight；还原共享状态列表",r("messages","states_from_client_messages"),code="B02")
    s.message("utils","server","states = [θ₁, …, θK]",code="B03",ret=True)
    s.end(opt)
    s.message("server","server","prepare_states(uploads)","校验 keys、shape、dtype、有限值；detach / clone 到服务器设备",r("response_server","ResponsePreservingServer.prepare_states"),code="B04")
    s.message("server","utils","distribution(...); mix_parameters(states, baseline, names)","baseline 和参考权重 ρ 默认均匀；先形成固定平均 θ_base",r("response_server","distribution"),code="B05")
    mode=s.fragment("alt","mode == 'fedavg'",left=60)
    s.message("server","entry","return θ_base, report", "optimization_steps=0；不创建探针；本分支结束",r("response_server","ResponsePreservingServer.execute"),code="B06",ret=True)
    s.alternative("mode ∈ {'response', 'output_only'}")
    s.message("server","probe","sample_probe_bank(... calibration_graphs, seed)","随后调用 diagnostic_graphs，seed+500009；seed=cfg.seed+round_id·1000003",r("server_probes","sample_probe_bank"),code="B07")
    s.message("probe","probe","SBM 图 → 高斯 x → structural_relations → pairs / directions", "共享空间输入 x[N,d]；关系 R[N,N,3]；方向 Δ[D,N,N]",r("server_probes","sample_probe_bank"),code="B08")
    s.message("probe","server","train / diagnostic 两套独立探针库", "train 用于优化；diagnostic 仅用于报告",r("server_probes","RelationProbe"),code="B09",ret=True)
    s.message("server","server","reference_cache(states, train)","随后 reference_cache(states, diagnostic)；均在 no_grad 下",r("response_server","ResponsePreservingServer.reference_cache"),code="B10")
    loop=s.fragment("loop","每个客户端 θᵢ、每个探针；先 train 库，再 diagnostic 库",left=560)
    s.message("server","eval","ref B-E：evaluate_state(θᵢ, bank)","functional_call 注入 θᵢ；参考值只计算一次并缓存",r("response_server","ResponsePreservingServer.evaluate_state"),code="B11")
    s.message("eval","server","reference outputs / responses", "堆叠为 [K,G,P] / [K,G,D,P]（响应不可用时 D=0）",r("response_server","ResponsePreservingServer.reference_cache"),code="B12",ret=True)
    s.end(loop)
    s.note("参考缓存与候选的差别", "参考与 diagnostics 默认在 response_available=True 时测响应；output_only 的候选评估显式 measure_response=False。不同 bank 各自有 G 个探针，P 为采样节点对数，D 为扰动方向数。",height=85)
    s.message("server","optim","Adam([logits], lr=cfg.lr)","logits = log(baseline)，requires_grad=True",r("response_server","ResponsePreservingServer.execute"),code="B13")
    loop=s.fragment("loop","step = 0 … config.steps−1；只更新聚合 logits",left=560)
    s.message("server","optim","zero_grad(set_to_none=True)",ref=r("response_server","ResponsePreservingServer.execute"),code="B14")
    s.message("server","server","objective(logits, states, train, refs, baseline, ρ)","w = softmax(logits) ∈ R^K，wᵢ≥0 且和为 1",r("response_server","ResponsePreservingServer.objective"),code="B15")
    s.message("server","utils","mix_parameters(states, w, parameter_names)","每个参数张量 θ(w)=Σwᵢθᵢ；保持可微计算图",r("response_server","mix_parameters"),code="B16")
    s.message("utils","server","候选 state θ(w)", "共享 buffers 必须相同，直接沿用",r("response_server","mix_parameters"),code="B17",ret=True)
    s.message("server","eval","ref B-E：evaluate_state(θ(w), train, measure_response)","response 模式测中心差分；output_only 仅测基准输出",r("response_server","ResponsePreservingServer.evaluate_state"),code="B18")
    s.message("eval","server","candidate outputs [G,P] / responses [G,D,P]", "梯度可回传到 w 与 logits",code="B19",ret=True)
    s.message("server","server","matching_terms(candidate, refs, ρ)","分别计算各客户端 MSE，再按固定 ρ 加权求和",r("response_server","ResponsePreservingServer.matching_terms"),code="B20")
    s.message("server","server","loss = λout·Lout + λprior·||w−baseline||²", "response 模式另加 λresponse·Lresponse",r("response_server","ResponsePreservingServer.objective"),code="B21")
    s.message("server","server","loss.backward(); 检查 logits.grad", "梯度经候选模型的 functional_call 与参数凸组合回传",r("response_server","ResponsePreservingServer.execute"),code="B22")
    s.message("server","optim","step()", "只改变 logits；states 与参考缓存不变",r("response_server","ResponsePreservingServer.execute"),code="B23")
    s.end(loop)
    s.message("server","utils","weights=softmax(logits); mix_parameters(...)","使用最后一个配置步得到的权重形成 final",r("response_server","mix_parameters"),code="B24")
    s.message("server","server","diagnostics(base/train) → diagnostics(final/train)","随后 base/diagnostic → final/diagnostic；复用 ref B-E",r("response_server","ResponsePreservingServer.diagnostics"),code="B25")
    s.message("server","entry","return final.detach().cpu(), report", "同时保存 last_probe_banks；诊断探针不用于挑选权重",r("response_server","ResponsePreservingServer.execute"),code="B26",ret=True)
    s.end(mode)
    s.message("entry","entry","save_aggregation(folder, state, ..., last_probe_banks)","写 shared_backbone.pt / aggregation_report.json / run_metadata.json；有探针时另写 server_probes.pt",r("server_io","save_aggregation"),code="B27")
    s.note("连接下一阶段", "联邦多轮：本轮 global_state 在下一轮经 named_weights() 下发给 A。阶段 C 的 load_shared() 读取 B 写出的 schema_version=3、stage='server_aggregation' 检查点。", "green",height=81)

    s.section("ref B-E｜evaluate_state → ProbeEvaluator.forward：每个探针如何评估",[
        ("caller","ResponsePreservingServer","evaluate_state / bank 堆叠","blue",r("response_server","ResponsePreservingServer.evaluate_state")),
        ("eval","ProbeEvaluator","forward(probe, ε, measure_response)","purple",r("server_probes","ProbeEvaluator.forward")),
        ("shared","SharedBackbone","仅调用 encode_experts","purple",r("model","SharedBackbone.encode_experts")),
        ("composer","RelationComposer","复用专家表示；仅扰动 relation_gate","purple",r("composer","RelationComposer.forward")),
        ("read","pair_readout","节点对余弦相似度","green",r("server_probes","pair_readout")),
    ],"N=探针节点数，M=专家数，d=隐藏维度，P=节点对数，D=扰动方向数。基准、正扰动和负扰动共享同一次专家编码 h。")
    loop=s.fragment("loop","probe in bank；每次调用注入当前候选 state")
    s.message("caller","eval","functional_call(evaluator, {'shared.'+k: tensor}, ...)","严格注入参数，不 load_state_dict，不切断候选梯度",r("response_server","ResponsePreservingServer.evaluate_state"),code="BE01")
    s.message("eval","shared","encode_experts(probe.x, probe.edge_index, None)","每个候选 / 探针只执行一次专家编码",r("model","SharedBackbone.encode_experts"),code="BE02")
    s.message("shared","eval","h[N,M,d]",code="BE03",ret=True)
    s.message("eval","composer","forward(h, R, relation_gate=ones[N,N])", "基准 gate G₀=1",r("composer","RelationComposer.forward"),code="BE04")
    s.message("composer","eval","z₀[N,d]",code="BE05",ret=True)
    s.message("eval","read","pair_readout(z₀, pairs[P,2])", "节点向量 L2 归一化，再算每对内积",r("server_probes","pair_readout"),code="BE06")
    s.message("read","eval","output[P]", "余弦相似度；不经过任务分类头",code="BE07",ret=True)
    opt=s.fragment("opt","measure_response == True",left=640)
    direction=s.fragment("loop","delta in probe.directions[D,N,N]",left=670)
    s.message("eval","composer","forward(h, R, relation_gate=1+εΔ)","只改变关系偏置 gate；专家 h 与图边保持不变",r("server_probes","ProbeEvaluator.forward"),code="BE08")
    s.message("composer","eval","positive[N,d]",code="BE09",ret=True)
    s.message("eval","composer","forward(h, R, relation_gate=1−εΔ)",ref=r("server_probes","ProbeEvaluator.forward"),code="BE10")
    s.message("composer","eval","negative[N,d]",code="BE11",ret=True)
    s.message("eval","read","pair_readout(positive, pairs); pair_readout(negative, pairs)",ref=r("server_probes","pair_readout"),code="BE12")
    s.message("read","eval","s₊[P], s₋[P]",code="BE13",ret=True)
    s.message("eval","eval","response_Δ = (s₊−s₋)/(2ε)","中心差分，测关系条件组合对扰动的敏感度",r("server_probes","ProbeEvaluator.forward"),code="BE14")
    s.end(direction)
    s.end(opt)
    s.message("eval","caller","output[P], response[D,P]", "measure_response=False 时 response[0,P]",r("server_probes","ProbeEvaluator.forward"),code="BE15",ret=True)
    s.end(loop)
    s.message("caller","caller","torch.stack(outputs), torch.stack(responses)","得到 [G,P]、[G,D,P]；返回 reference_cache / objective / diagnostics",r("response_server","ResponsePreservingServer.evaluate_state"),code="BE16")
    s.finish("先读 B01–B27；B11/B18/B25 引用下半页 B-E。注意 B 优化的是客户端聚合系数，候选网络参数是它们的可微组合；服务器无需再次询问客户端。")
    return s


def make_c(mx):
    s=Sequence(mx,"c","阶段 C｜目标域本地微调 · 代码时序图",
               "上半页：加载、适配安装、训练 / 验证 / 测试。下半页：一次 TargetModel.forward 的精确顺序。",
               "输入：B 的聚合共享检查点、目标域 train / valid / test；输出：验证集选择的 target_model.pt 与最终测试指标。三种 adaptation 互斥。")
    s.section("C-1｜run_target：初始化与训练调度",[
        ("run","finetune.main → run_target","目标域训练控制","gray",r("target_train","run_target")),
        ("load","加载函数","load_target_manifest / load_shared","green",r("target_model","load_shared")),
        ("model","TargetModel","冻结基座、安装适配、前向与参数审计","purple",r("target_model","TargetModel")),
        ("optim","torch.optim.AdamW","只接收 requires_grad=True 参数","yellow",r("target_train","run_target")),
    ],f"入口：finetune_fedg2c.py:{source('finetune_fedg2c.py','main')[2]}，启用 --use-new-design → fedg2c/target_train.py:{r('target_train','run_target')[2]}；默认 adaptation='lora'、local_alpha=0、learnable_alpha=False。")
    s.message("run","load","load_target_manifest(manifest_path, config)","显式校验划分；图任务保留整图，节点任务取局部图并重映射 root",r("target_data","load_target_manifest"),code="C01")
    s.message("load","run","manifest, splits{'train','valid','test'}", "每个 TargetSample 含 view、label、root、sample_id",r("target_data","TargetSample"),code="C02",ret=True)
    s.message("run","load","load_shared(checkpoint)","校验 B 阶段、schema、relation_schema 和 state_digest；构造并加载 SharedBackbone",r("target_model","load_shared"),code="C03")
    s.message("load","run","shared, shared_config, metadata",code="C04",ret=True)
    s.message("run","model","TargetModel(shared, shared_config, feature_schema, task, config)",ref=r("target_model","TargetModel.__init__"),code="C05")
    s.message("model","model","shared.requires_grad_(False).eval(); 记录原基座摘要", "新建目标域 node/edge adapter、分类 head 和可选 local_encoder / alpha_logit",r("target_model","TargetModel.__init__"),code="C06")
    mode=s.fragment("alt","adaptation == 'lora'",left=1850)
    s.message("model","model","install_lora(shared.composer, config)","仅替换 query / key 为 LoRALinear；复制并冻结原 W/b，merge_weights=False",r("target_model","install_lora"),code="C07")
    s.alternative("adaptation == 'adapter'",left=1850)
    s.message("model","model","ResidualAdapter(dim=d, bottleneck=b)","down: d→b，up: b→d；up 权重和偏置初始化为零",r("target_model","ResidualAdapter.__init__"),code="C08")
    s.alternative("adaptation == 'prompt'",left=1850)
    s.message("model","model","prompt = nn.Parameter(zeros(1,1,d))", "在专家 token 空间共享一条可学习向量",r("target_model","TargetModel.__init__"),code="C09")
    s.alternative("adaptation == 'none'",left=1850)
    s.message("model","model","保留共同私有模块；不安装上述三种适配",ref=r("target_model","TargetModel.__init__"),code="C10")
    s.end(mode)
    s.note("纯本地对照分支", "若 local_alpha=1 且 learnable_alpha=False：再次冻结共享部分（含已安装 LoRA）以及 Prompt / Adapter；前向跳过共享专家和组合器，只用 local_encoder。默认配置不进入该分支。",height=82)
    s.message("run","optim","AdamW([p for p in model.parameters() if p.requires_grad])", "训练参数见 TargetModel.parameter_report()；每次 run_target 新建优化器",r("target_train","run_target"),code="C11")
    epochs=s.fragment("loop","epoch = 0 … config.epochs−1")
    s.message("run","model","train()", "内部 super().train() 后把 self.shared 设回 eval；仍允许梯度穿过冻结基座",r("target_model","TargetModel.train"),code="C12")
    s.message("run","run","torch.randperm(train_size)","打乱训练样本顺序，再按 batch_size 分组",r("target_train","run_target"),code="C13")
    batch=s.fragment("loop","每个训练 mini-batch")
    s.message("run","model","predict(model, samples, device) → ref C-F", "predict 内逐样本调用 model(view,root)，最后 torch.cat；不是图拼接 batch",r("target_train","predict"),code="C14")
    s.message("model","run","逐样本 logits[1,O]；predict 拼成 logits[B,O]", "labels[B,1]（多分类）或 labels[B,O]（二分类 / 多标签）",r("target_train","predict"),code="C15",ret=True)
    s.message("run","run","classification_loss(logits, labels, objective)","multiclass：cross_entropy；其他：有限标签 mask + BCEWithLogits；count=0 则跳过该 batch",r("target_train","classification_loss"),code="C16")
    observed=s.fragment("opt","count > 0；否则 continue 到下一 mini-batch")
    s.message("run","optim","zero_grad(set_to_none=True)", "源码位置在前向和 loss 计算之后",r("target_train","run_target"),code="C17")
    s.message("run","run","loss.backward() → clip_grad_norm_(trainable, 5)","梯度穿过冻结算子，只有 requires_grad=True 的参数累积梯度",r("target_train","run_target"),code="C18")
    s.message("run","optim","step()", "更新本地私有模块和当前适配参数",r("target_train","run_target"),code="C19")
    s.end(observed)
    s.end(batch)
    s.message("run","run","evaluate(model, valid, device)", "@torch.no_grad()；先 model.eval()，再 predict → metrics",r("target_train","evaluate"),code="C20")
    s.message("run","model","验证集逐样本 forward：ref C-F", "读取 valid；不更新模型参数",r("target_train","predict"),code="C21")
    s.message("model","run","valid logits → metrics → valid['loss']", "metrics 在 evaluate 内计算",r("target_train","metrics"),code="C22",ret=True)
    best=s.fragment("opt","valid['loss'] < best_loss；相同 loss 保留较早 epoch")
    s.message("run","run","best_state = clone(model.state_dict())", "保存最小验证损失对应的权重与 best_epoch",r("target_train","run_target"),code="C23")
    s.end(best)
    s.end(epochs)
    s.message("run","model","load_state_dict(best_state); frozen_base_unchanged()", "先恢复最佳验证检查点，再核验原共享基座摘要",r("target_model","TargetModel.frozen_base_unchanged"),code="C24")
    s.message("run","run","evaluate(model, test, device)", "选择完成后才运行一次最终测试；复用 ref C-F",r("target_train","evaluate"),code="C25")
    s.message("run","model","parameter_report(); alpha()", "统计本地可训练参数与实际融合系数",r("target_model","TargetModel.parameter_report"),code="C26")
    s.message("run","run","torch.save(best_state)；写 report / metadata", "输出 target_model.pt、metrics.json、run_metadata.json；run_target 返回 report",r("target_train","run_target"),code="C27")

    s.section("ref C-F｜TargetModel.forward：三种适配插入点与可选本地融合",[
        ("call","predict / evaluate","单个 TargetSample","gray",r("target_train","predict")),
        ("model","TargetModel","forward(view,root)","blue",r("target_model","TargetModel.forward")),
        ("feat","FeatureAdapter","目标域私有 node / edge adapter","yellow",r("model","FeatureAdapter.forward")),
        ("shared","SharedBackbone","冻结专家；encode_experts","purple",r("model","SharedBackbone.encode_experts")),
        ("comp","RelationComposer","冻结基座 + 可选 Q/K LoRA","purple",r("composer","RelationComposer.forward")),
        ("adapt","ResidualAdapter","仅 adaptation='adapter' 时存在","yellow",r("target_model","ResidualAdapter.forward")),
        ("local","local_encoder","可选本地 EdgePassingGraphEncoder","yellow",r("model","EdgePassingGraphEncoder.encode")),
        ("head","head: nn.Linear(d,O)","任务分类头；三个方案均训练","yellow",r("target_model","TargetModel.forward")),
    ],"N=该样本局部节点数，M=专家数，d=隐藏维度，O=task.num_outputs；B=当前 mini-batch 样本数。默认 local_alpha=0，只走共享表示适配分支。")
    s.message("call","model","forward(view, root=None / local_root)",ref=r("target_model","TargetModel.forward"),code="CF01")
    s.message("model","feat","node_adapter(view.x[N,F]); edge_adapter(edge_attr)?", "离散列 embedding 求和或连续 Linear；边特征由 schema 显式控制",r("model","FeatureAdapter.forward"),code="CF02")
    s.message("feat","model","x[N,d]；可选 edge[E,d]",code="CF03",ret=True)
    shared=s.fragment("opt","not pure_local；pure_local=(local_alpha==1 且 alpha_logit is None)",left=430)
    s.message("model","shared","encode_experts(x, edges, edge)", "M 个冻结图专家依次前向；内部细节同 A-S 的专家部分",r("model","SharedBackbone.encode_experts"),code="CF04")
    s.message("shared","model","h[N,M,d]",code="CF05",ret=True)
    prompt=s.fragment("opt","adaptation == 'prompt'",left=440,right=1640)
    s.message("model","model","h = h + self.prompt[1,1,d]", "广播到所有节点、所有专家 token",r("target_model","TargetModel.forward"),code="CF06")
    s.end(prompt)
    s.message("model","model","structural_relations(view.edge_index,N)","重新构造当前目标样本 R[N,N,3]",r("context","structural_relations"),code="CF07")
    s.message("model","comp","forward(h, relations)", "这里直接调用 composer；不再经过 SharedBackbone.forward",r("composer","RelationComposer.forward"),code="CF08")
    lora=s.fragment("opt","adaptation == 'lora'：composer.query / key 内部",left=1630,right=2440)
    s.message("comp","comp","LoRALinear.forward(tokens[N·M,d])", "Q/K = tokens·Wᵀ + b + (lora_alpha/r)·tokens·Aᵀ·Bᵀ",r("vendor/loralib/layers","Linear.forward"),code="CF09")
    s.end(lora)
    s.note("组合器其余运算", "RelationComposer.forward：Q/K/V → 加关系偏置 → head_rms + hypernetwork_contract（默认）→ output + 残差 + LayerNorm → 对 M 个专家取均值。Prompt 梯度穿过整个冻结组合器；LoRA 梯度回传到 Q/K 的 A、B。",height=91,left=1270,width=1910)
    s.message("comp","model","z[N,d]", "普通 / Prompt / LoRA 模式的组合表示",code="CF10",ret=True)
    adapter=s.fragment("opt","adaptation == 'adapter'",left=440,right=2440)
    s.message("model","adapt","ResidualAdapter.forward(z[N,d])", "z + up(ReLU(down(z)))；瓶颈 [N,b]",r("target_model","ResidualAdapter.forward"),code="CF11")
    s.message("adapt","model","z_adapted[N,d]", "Adapter 位于组合器之后",code="CF12",ret=True)
    s.end(adapter)
    s.end(shared)
    local=s.fragment("opt","local_encoder is not None（local_alpha>0 或 learnable_alpha）",left=430)
    s.message("model","local","GraphEncoder.forward(x, edges, edge) → encode(...) ", "新的可训练本地编码器，不是阶段 A 的动量分支",r("model","EdgePassingGraphEncoder.encode"),code="CF13")
    s.message("local","model","local[N,d]",code="CF14",ret=True)
    s.message("model","model","融合：普通分支调用 alpha()", "普通：z=α·local+(1−α)·z；pure_local：直接 z=local",r("target_model","TargetModel.alpha"),code="CF15")
    s.end(local)
    readout=s.fragment("alt","task.level == 'graph'",left=430,right=1640)
    s.message("model","model","z.mean(dim=0, keepdim=True)","全部节点均值 → [1,d]",r("target_model","TargetModel.forward"),code="CF16")
    s.alternative("task.level == 'node'",left=430,right=1640)
    s.message("model","model","z[root:root+1]", "只读重映射后的目标根节点 → [1,d]",r("target_model","TargetModel.forward"),code="CF17")
    s.end(readout)
    s.message("model","head","head(z[1,d])", "nn.Linear：d → O；前向返回原始 logits",r("target_model","TargetModel.forward"),code="CF18")
    s.message("head","call","logits[1,O]（经 TargetModel 返回）", "predict 收集 B 个样本后拼接，再在外部计算监督 loss",r("target_train","predict"),code="CF19",ret=True)
    s.note("可训练参数的共同部分", "node_adapter、可选 edge_adapter、head；可选 local_encoder / alpha_logit。除此以外，Prompt 训练 prompt；LoRA 训练 Q/K 的 lora_A、lora_B；Adapter 训练 down/up。原始共享专家和组合器参数冻结。", "green",height=88)
    s.finish("先读 C01–C27，再看 C-F。训练和验证使用同一个 forward，区别由 model.train()/eval() 与 no_grad 决定；shared.eval() 只固定模式，不能替代参数冻结或 no_grad。")
    return s


def main():
    mx=ET.Element("mxfile",host="Electron",agent="FedG2C source sequence diagrams",version="29.7.9",pages="3")
    pages=[make_a(mx),make_b(mx),make_c(mx)]
    ET.indent(mx,space="  ")
    OUTPUT.write_bytes(ET.tostring(mx,encoding="utf-8",xml_declaration=True))
    # Structural and source-location validation, without importing training code.
    parsed=ET.parse(OUTPUT)
    assert len(parsed.findall("diagram"))==3
    cells=0
    for diagram in parsed.findall("diagram"):
        root=diagram.find("mxGraphModel/root")
        ids=[c.get("id") for c in root]
        assert len(ids)==len(set(ids))
        assert "0" in ids and "1" in ids
        for cell in root:
            if cell.get("edge")=="1":
                assert cell.find("mxGeometry") is not None
                assert len(cell.findall("mxGeometry/mxPoint"))==2
            if cell.get("vertex")=="1":
                geo=cell.find("mxGeometry")
                assert float(geo.get("width"))>0 and float(geo.get("height"))>0
                assert float(geo.get("x"))>=0
        cells+=len(ids)
    assert b"<!--" not in OUTPUT.read_bytes()
    (OUTPUT.parent/"fedg2c-sequence-diagrams"/"source-index.json").write_text(
        json.dumps({"source_references":[{"path":p,"symbol":sym,"line":ln} for p,sym,ln in REFS],
                    "pages":[{"stage":s.key,"height":s.y,"messages":s.message_records} for s in pages]},ensure_ascii=False,indent=2),encoding="utf-8")
    guide=[
        "# FedG2C 阶段 A / B / C 代码时序图", "",
        "打开 [fedg2c-stage-sequences.drawio](fedg2c-stage-sequences.drawio)，底部有 A、B、C 三个页面。每页上半部分为调度顺序，下半部分展开关键调用。", "",
        "PNG 预览：[阶段 A](fedg2c-sequence-diagrams/preview-a.png) · [阶段 B](fedg2c-sequence-diagrams/preview-b.png) · [阶段 C](fedg2c-sequence-diagrams/preview-c.png)。可编辑源文件以 drawio 为准。", "",
        "图中实线是调用，虚线是返回，折返箭头表示对象内部执行（或在本对象内调用标注的工具函数）；loop / alt / opt 表示循环、互斥分支和条件步骤。蓝色/紫色表示共享基座及参考模块，黄色表示本地可训练模块或优化器，绿色表示数据/结构函数。各页阶段标记按执行顺序编号。", "",
        "源码位置采用 `相对路径:行号`，指向当前文件中的实际定义；同一函数中多个操作共享同一定义位置。鼠标悬停可查看完整类/方法名。对象生命线有时合并同类实例或工具函数组，已在参与者名称和旁注中说明；这种合并不表示实例共享权重。", "",
        "## A：本地预训练", "",
        "- A01–A07：建模、加载本轮共享权重、重置动量分支、创建本地优化器。",
        "- A08–A28：采样双视图 → 可见图主分支 → 重算位置 → GraphSAGE/Transformer 预测 → 完整局部图动量参考 → Smooth L1。",
        "- A29–A35：backward → clip → optimizer.step → EMA → 导出共享权重。",
        "- AS01–AS14：专家逐一编码 → 结构关系 → 关系条件组合。", "",
        "独立 A 入口：`pretrain_fedg2c.main --use-new-design` → `fedg2c.edge_entry.run`。联邦入口：`pretrain_federated_fedg2c.main` 在每轮、每个客户端执行 A，再执行 B。", "",
        "## B：服务器聚合", "",
        "- B01–B09：解包与校验；fedavg 提前返回；其他模式生成训练/诊断探针。",
        "- B10–B23：缓存客户端参考 → 学习 logits → 可微混合候选参数 → 输出/响应匹配 → 更新聚合系数。",
        "- B24–B27：最后一个优化步的聚合权重 → 独立诊断 → 保存 B 检查点。",
        "- BE01–BE16：functional_call → 一次专家编码 → 基准组合输出 → 每个方向正负 gate 扰动 → 节点对余弦中心差分。", "",
        "B 的基准权重和参考权重默认均匀，消息中的 num_samples=1 是兼容字段。output_only 的候选不测响应；参考缓存和诊断在关系分支可用时仍可测响应。扰动仅作用于 composer 的 relation_gate，专家编码和图边不变。", "",
        "## C：目标域适配", "",
        "- C01–C11：加载显式数据划分和 B 检查点 → 冻结原基座 → 安装三种适配之一 → 创建本地优化器。",
        "- C12–C23：按 epoch/batch 训练；每个 batch 逐样本前向后拼接 logits；每个 epoch 验证并选择最低验证 loss。",
        "- C24–C27：恢复最佳权重 → 校验冻结基座 → 一次最终测试 → 保存目标模型。",
        "- CF01–CF19：私有输入适配 → 冻结专家 → 可选 Prompt → 关系组合器（可选 Q/K LoRA）→ 可选输出 Adapter → 可选本地编码融合 → 图/节点读出 → head。", "",
        "| 方案 | 插入位置 | 额外训练参数 |", "|---|---|---|",
        "| Prompt | 专家输出 h[N,M,d] 加广播 prompt[1,1,d]，随后进入组合器 | prompt |",
        "| LoRA | 组合器 query/key 的 Linear.forward 内 | Q/K 各自的 lora_A[r,d]、lora_B[d,r] |",
        "| Adapter | 组合器输出 z[N,d] 之后 | down: d→b；up: b→d |", "",
        "三种方案都保留目标域私有 node/edge adapter、分类 head，以及配置启用的 local_encoder / alpha_logit。默认 local_alpha=0。固定 local_alpha=1 为纯本地对照，跳过共享基座并冻结闲置的适配参数。", "",
        "## 源码索引", "",
    ]
    for p,sym,ln in sorted(REFS):
        guide.append(f"- `{sym}` — [{p}:{ln}](../{p}#L{ln})")
    guide += ["", "生成器：[generate.py](fedg2c-sequence-diagrams/generate.py)。运行后会按 AST 重新解析当前源码的类/函数行号，并检查 XML、唯一 ID、边几何和页面数量。只读源码，不修改训练实现。", ""]
    (OUTPUT.parent/"fedg2c-stage-sequences-guide.md").write_text("\n".join(guide),encoding="utf-8")
    print(json.dumps({"output":str(OUTPUT),"pages":[{"name":s.key,"messages":len(s.message_records),"height":s.y} for s in pages],"cells":cells,"source_references":len(REFS)},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
