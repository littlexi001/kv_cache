from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the inline length/PPL mechanism visualization.")
    parser.add_argument("--analysis_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    summary = json.loads((analysis_dir / "mechanism_summary.json").read_text(encoding="utf-8"))
    with (analysis_dir / "length_mechanism_rows.csv").open(encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    points = [
        {
            "length": int(row["length"]),
            "ppl": round(float(row["gold_ppl"]), 6),
            "mass": round(float(row["mass.hop2_result"]), 10),
            "logit": round(float(row["hop2_result.mean_logit"]), 6),
            "lse": round(float(row["mean_head_logsumexp"]), 6),
        }
        for row in raw_rows
    ]
    payload = {
        "points": points,
        "bins": summary["binned"],
        "decomposition": summary["attention_decomposition"]["hop2_result"],
        "layers": summary["hop2_result_layer_diagnostics"],
        "shortPpl": summary["short_ppl"]["median"],
        "longPpl": summary["long_ppl"]["median"],
        "pplFactor": summary["median_ppl_factor_long_over_short"],
        "massCorrelation": summary["correlations_with_log_ppl"]["mass.hop2_result"],
        "concentration": summary["hop2_result_head_concentration"],
    }
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    fragment = f"""<div id="length-ppl-mechanism" aria-label="长上下文正确答案 PPL 退化机制诊断">
  <div class="viz-grid lpm-stats">
    <div class="card viz-stat"><span class="text-muted">Gold PPL 中位数</span><span class="viz-stat-value">13.05 → 2382</span><span class="text-small text-muted">短 ≤8K；长 ≥120K</span></div>
    <div class="card viz-stat"><span class="text-muted">最终证据 mass 相关</span><span class="viz-stat-value">ρ = −0.913</span><span class="text-small text-muted">与 log(PPL) 的 Spearman</span></div>
    <div class="card viz-stat"><span class="text-muted">几何 attention 保留</span><span class="viz-stat-value">0.260%</span><span class="text-small text-muted">方向与分母共同作用</span></div>
  </div>
  <section class="lpm-section">
    <h3>Gold PPL 随上下文长度</h3>
    <svg id="lpm-ppl" role="img" aria-label="257 个长度点的 Gold PPL 对数曲线"></svg>
  </section>
  <section class="lpm-section">
    <h3>最终证据 attention 的双重衰减</h3>
    <div id="lpm-decomposition" role="img" aria-label="方向退化与 softmax 竞争对 attention 衰减的贡献"></div>
  </section>
  <section class="lpm-section lpm-two">
    <div>
      <h3>最终证据 mass 与答案 PPL</h3>
      <svg id="lpm-scatter" role="img" aria-label="basket attention mass 与 Gold PPL 的散点图"></svg>
    </div>
    <div>
      <h3>各层最终证据 mass 与 PPL 的相关</h3>
      <div id="lpm-layers" class="lpm-layer-grid" role="img" aria-label="36 层的 basket mass 与 PPL Spearman 相关热力图"></div>
      <div class="lpm-scale text-small"><span>弱</span><i></i><span>强负相关</span></div>
    </div>
  </section>
  <span class="sr-only">上下文越长，最终证据 raw logit 下降且 softmax 分母上升；两者共同降低证据 attention，并与正确答案 PPL 上升高度相关。</span>
</div>
<style>
  #length-ppl-mechanism {{ width:100%; color:var(--foreground); }}
  #length-ppl-mechanism .lpm-stats {{ margin-bottom:1rem; }}
  #length-ppl-mechanism .viz-stat {{ display:flex; flex-direction:column; gap:.2rem; }}
  #length-ppl-mechanism .lpm-section {{ margin:1rem 0 1.4rem; }}
  #length-ppl-mechanism h3 {{ margin:0 0 .5rem; font-weight:500; }}
  #length-ppl-mechanism svg {{ display:block; width:100%; height:auto; overflow:visible; }}
  #length-ppl-mechanism .lpm-two {{ display:grid; grid-template-columns:minmax(0,1.25fr) minmax(260px,.75fr); gap:1.25rem; align-items:start; }}
  #length-ppl-mechanism .axis {{ stroke:var(--border); stroke-width:1; }}
  #length-ppl-mechanism .gridline {{ stroke:var(--border); stroke-width:1; opacity:.55; }}
  #length-ppl-mechanism .tick {{ fill:var(--muted-foreground); font-size:11px; }}
  #length-ppl-mechanism .raw-point {{ fill:var(--muted-foreground); opacity:.28; }}
  #length-ppl-mechanism .median-line {{ fill:none; stroke:var(--viz-series-1); stroke-width:2.5; }}
  #length-ppl-mechanism .median-point {{ fill:var(--viz-series-1); stroke:var(--background); stroke-width:1.5; }}
  #length-ppl-mechanism .mass-point {{ fill:var(--viz-series-2); opacity:.42; }}
  #length-ppl-mechanism .lpm-decomp-track {{ display:flex; min-height:54px; overflow:hidden; background:var(--muted); }}
  #length-ppl-mechanism .lpm-decomp-part {{ display:flex; flex-direction:column; justify-content:center; padding:.55rem .7rem; color:var(--foreground); min-width:0; }}
  #length-ppl-mechanism .lpm-decomp-part b {{ font-weight:500; }}
  #length-ppl-mechanism .lpm-decomp-part span {{ font-size:11px; }}
  #length-ppl-mechanism .direction {{ background:color-mix(in srgb, var(--viz-series-1) 34%, transparent); }}
  #length-ppl-mechanism .competition {{ background:color-mix(in srgb, var(--viz-series-2) 34%, transparent); }}
  #length-ppl-mechanism .lpm-decomp-total {{ margin-top:.45rem; display:flex; justify-content:space-between; gap:.8rem; color:var(--muted-foreground); }}
  #length-ppl-mechanism .lpm-layer-grid {{ display:grid; grid-template-columns:repeat(9,minmax(28px,1fr)); gap:4px; }}
  #length-ppl-mechanism .lpm-layer-cell {{ display:flex; align-items:center; justify-content:center; aspect-ratio:1; background:color-mix(in srgb, var(--viz-series-3) calc(var(--strength) * 100%), var(--muted)); font-size:11px; }}
  #length-ppl-mechanism .lpm-scale {{ display:flex; align-items:center; gap:.45rem; margin-top:.45rem; color:var(--muted-foreground); }}
  #length-ppl-mechanism .lpm-scale i {{ height:8px; flex:1; background:linear-gradient(90deg,var(--muted),var(--viz-series-3)); }}
  @media (max-width:620px) {{ #length-ppl-mechanism .lpm-two {{ grid-template-columns:1fr; }} #length-ppl-mechanism .lpm-layer-grid {{ grid-template-columns:repeat(6,minmax(28px,1fr)); }} }}
</style>
<script>
(() => {{
  const root = document.getElementById("length-ppl-mechanism");
  const data = {data_json};
  const NS = "http://www.w3.org/2000/svg";
  const el = (name, attrs={{}}, text="") => {{ const node=document.createElementNS(NS,name); Object.entries(attrs).forEach(([k,v])=>node.setAttribute(k,String(v))); if(text) node.textContent=text; return node; }};
  const log10 = value => Math.log(value) / Math.LN10;
  const drawAxes = (svg, width, height, margin, xTicks, yTicks, xMap, yMap, xFormat, yFormat) => {{
    yTicks.forEach(value => {{ const y=yMap(value); svg.append(el("line",{{x1:margin.left,y1:y,x2:width-margin.right,y2:y,class:"gridline"}})); svg.append(el("text",{{x:margin.left-7,y:y+4,"text-anchor":"end",class:"tick"}},yFormat(value))); }});
    xTicks.forEach(value => {{ const x=xMap(value); svg.append(el("line",{{x1:x,y1:height-margin.bottom,x2:x,y2:height-margin.bottom+4,class:"axis"}})); svg.append(el("text",{{x,y:height-5,"text-anchor":"middle",class:"tick"}},xFormat(value))); }});
    svg.append(el("line",{{x1:margin.left,y1:height-margin.bottom,x2:width-margin.right,y2:height-margin.bottom,class:"axis"}}));
    svg.append(el("line",{{x1:margin.left,y1:margin.top,x2:margin.left,y2:height-margin.bottom,class:"axis"}}));
  }};
  const drawPpl = () => {{
    const svg=root.querySelector("#lpm-ppl"), width=720, height=245, margin={{left:55,right:18,top:12,bottom:28}}; svg.setAttribute("viewBox",`0 0 ${{width}} ${{height}}`);
    svg.append(el("title",{{}},"Gold PPL 随长度变化")); svg.append(el("desc",{{}},"灰点是 257 个长度点，蓝线是区间中位数，纵轴为对数尺度。"));
    const yMin=0, yMax=5; const xMap=v=>margin.left+(v/128000)*(width-margin.left-margin.right); const yMap=v=>height-margin.bottom-((log10(v)-yMin)/(yMax-yMin))*(height-margin.top-margin.bottom);
    drawAxes(svg,width,height,margin,[0,32000,64000,96000,128000],[1,10,100,1000,10000,100000],xMap,yMap,v=>v===0?"0":`${{v/1000}}K`,v=>v>=1000?`${{v/1000}}K`:String(v));
    data.points.forEach(p=>{{ const c=el("circle",{{cx:xMap(p.length),cy:yMap(p.ppl),r:2,class:"raw-point"}}); c.append(el("title",{{}},`${{p.length.toLocaleString()}} tokens · PPL ${{p.ppl.toFixed(2)}}`)); svg.append(c); }});
    const med=data.bins.filter(b=>b.start>=1000).map(b=>({{x:(b.start+b.stop_exclusive-1)/2,y:b.ppl_median}}));
    svg.append(el("path",{{d:med.map((p,i)=>`${{i?"L":"M"}}${{xMap(p.x).toFixed(1)}},${{yMap(p.y).toFixed(1)}}`).join(" "),class:"median-line"}}));
    med.forEach(p=>{{ const c=el("circle",{{cx:xMap(p.x),cy:yMap(p.y),r:4,class:"median-point"}}); c.append(el("title",{{}},`区间中位数 PPL ${{p.y.toFixed(2)}}`)); svg.append(c); }});
  }};
  const drawDecomposition = () => {{
    const host=root.querySelector("#lpm-decomposition"), d=data.decomposition, total=-d.delta_geometric_log_mass;
    const track=document.createElement("div"); track.className="lpm-decomp-track";
    const direction=document.createElement("div"); direction.className="lpm-decomp-part direction"; direction.style.width=`${{d.direction_share_of_log_attenuation*100}}%`; direction.innerHTML=`<b>方向退化 51.4%</b><span>raw logit −3.063 · 分子 ×0.0467</span>`;
    const competition=document.createElement("div"); competition.className="lpm-decomp-part competition"; competition.style.width=`${{d.competition_share_of_log_attenuation*100}}%`; competition.innerHTML=`<b>softmax 竞争 48.6%</b><span>logsumexp +2.891 · 概率 ×0.0555</span>`;
    track.append(direction,competition); host.append(track);
    const footer=document.createElement("div"); footer.className="lpm-decomp-total text-small"; footer.innerHTML=`<span>总 log-attention 衰减 ${{total.toFixed(3)}} nats</span><span>几何 mass ×${{d.combined_geometric_mass_factor.toFixed(4)}}</span>`; host.append(footer);
  }};
  const drawScatter = () => {{
    const svg=root.querySelector("#lpm-scatter"), width=460, height=300, margin={{left:58,right:16,top:12,bottom:35}}; svg.setAttribute("viewBox",`0 0 ${{width}} ${{height}}`);
    svg.append(el("title",{{}},"basket attention mass 与 Gold PPL")); svg.append(el("desc",{{}},"basket mass 越低，正确答案 PPL 越高。Spearman 相关为负 0.913。"));
    const xs=data.points.map(p=>log10(p.mass)), ys=data.points.map(p=>log10(p.ppl)); const xmin=Math.floor(Math.min(...xs)), xmax=Math.ceil(Math.max(...xs)); const ymin=0,ymax=5;
    const xMap=v=>margin.left+((v-xmin)/(xmax-xmin))*(width-margin.left-margin.right); const yMap=v=>height-margin.bottom-((log10(v)-ymin)/(ymax-ymin))*(height-margin.top-margin.bottom);
    drawAxes(svg,width,height,margin,[xmin,xmin+1,xmin+2,xmax],[1,10,100,1000,10000,100000],xMap,yMap,v=>`10^${{v}}`,v=>v>=1000?`${{v/1000}}K`:String(v));
    data.points.forEach(p=>{{ const c=el("circle",{{cx:xMap(log10(p.mass)),cy:yMap(p.ppl),r:2.7,class:"mass-point"}}); c.append(el("title",{{}},`${{p.length.toLocaleString()}} tokens · mass ${{(p.mass*100).toFixed(4)}}% · PPL ${{p.ppl.toFixed(2)}}`)); svg.append(c); }});
    svg.append(el("text",{{x:width-margin.right,y:margin.top+12,"text-anchor":"end",class:"tick"}},"Spearman ρ = −0.913"));
    svg.append(el("text",{{x:(margin.left+width-margin.right)/2,y:height-3,"text-anchor":"middle",class:"tick"}},"basket attention mass（log10）"));
  }};
  const drawLayers = () => {{
    const host=root.querySelector("#lpm-layers"); data.layers.forEach(layer=>{{ const cell=document.createElement("div"); cell.className="lpm-layer-cell"; cell.style.setProperty("--strength",String(Math.max(0,Math.min(1,Math.abs(layer.mass_logppl_spearman))))); cell.textContent=`L${{layer.layer}}`; cell.setAttribute("aria-label",`Layer ${{layer.layer}}，mass 与 log PPL Spearman ${{layer.mass_logppl_spearman.toFixed(3)}}，长短 mass retention ${{(layer.mass_retention_factor*100).toFixed(1)}}%`); cell.setAttribute("data-tooltip",`ρ ${{layer.mass_logppl_spearman.toFixed(3)}} · retention ${{(layer.mass_retention_factor*100).toFixed(1)}}%`); host.append(cell); }});
  }};
  drawPpl(); drawDecomposition(); drawScatter(); drawLayers();
}})();
</script>
"""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(fragment, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
