import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "file:///C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const ROOT = process.env.ACP_ROOT;
const OUT = process.env.ACP_PPTX;
const QA = process.env.ACP_QA;
const W = 1280, H = 720;
const C = { ink: "#111111", muted: "#5f6670", panel: "#ededed", rule: "#b8bcc4", blue: "#3d8dff", cyan: "#6dcbf4", red: "#c44e52", green: "#2f8f5b", white: "#ffffff" };

async function writeBlob(file, blob) { await fs.writeFile(file, new Uint8Array(await blob.arrayBuffer())); }
function box(slide, x, y, w, h, fill=C.panel, line=C.panel) {
  return slide.shapes.add({ geometry:"rect", position:{left:x,top:y,width:w,height:h}, fill, line:{style:"solid",fill:line,width:1} });
}
function text(slide, value, x, y, w, h, size=24, opts={}) {
  const s=slide.shapes.add({geometry:"textbox",position:{left:x,top:y,width:w,height:h},fill:"none",line:{style:"solid",fill:"none",width:0}});
  s.text=value; s.text.style={fontFamily:"Arial",fontSize:size,color:opts.color||C.ink,bold:!!opts.bold,alignment:opts.align||"left",verticalAlignment:opts.valign||"middle"}; return s;
}
function title(slide, value, kicker) {
  if(kicker) text(slide,kicker.toUpperCase(),58,28,550,28,14,{bold:true,color:C.blue});
  text(slide,value,58,58,1164,78,38,{bold:true});
  box(slide,58,142,1164,2,C.rule,C.rule);
}
function footer(slide, n) { text(slide,String(n).padStart(2,"0"),1182,675,40,20,12,{color:C.muted,align:"right"}); }
function bullets(slide, items, x=72, y=180, w=520, size=23, gap=64) {
  items.forEach((v,i)=>{ box(slide,x,y+i*gap+10,9,9,C.blue,C.blue); text(slide,v,x+24,y+i*gap,w-24,48,size); });
}
async function image(slide, rel, x, y, w, h) {
  const p=path.join(ROOT,rel); const b=await fs.readFile(p);
  slide.images.add({blob:b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength),contentType:"image/png",alt:path.basename(rel),fit:"contain",position:{left:x,top:y,width:w,height:h}});
}
function newSlide(p,n){const s=p.slides.add(); s.background.fill=C.white; footer(s,n); return s;}

const p=Presentation.create({slideSize:{width:W,height:H}});

{
 const s=newSlide(p,1); text(s,"ACP-GINI",58,40,250,28,16,{bold:true,color:C.blue});
 text(s,"Ancestor-correlation penalties\nfor less redundant decision trees",58,150,790,190,54,{bold:true});
 box(s,910,110,250,250,C.ink,C.ink); text(s,"ΔG",935,140,200,90,62,{bold:true,color:C.white,align:"center"}); text(s,"×",990,225,90,55,42,{color:C.cyan,align:"center"}); text(s,"path penalty",935,288,200,45,24,{color:C.white,align:"center"});
 text(s,"Huynh Khang Lam  •  Tuan Dat Phan  •  Trong Nhan Nguyen\nFPT University",58,550,900,72,20,{color:C.muted});
}
{
 const s=newSlide(p,2); title(s,"Trees avoid coefficient failure — not redundant evidence","Problem");
 text(s,"X₁",120,225,150,90,64,{bold:true,align:"center"}); text(s,"≈",300,225,100,90,64,{color:C.blue,align:"center"}); text(s,"X₂",430,225,150,90,64,{bold:true,align:"center"});
 box(s,680,190,460,300,"#f5f5f5",C.rule); bullets(s,["Near-equal gains make representatives interchangeable","Several members of one correlated block can reappear downstream","Accuracy can stay strong while the explanation repeats itself"],710,210,400,20,82);
}
{
 const s=newSlide(p,3); title(s,"The practical cost is interpretability, not only prediction","Consequences");
 const labels=[["Unstable structure","small sample changes alter downstream branches"],["Diluted importance","one signal is divided across substitutes"],["Redundant paths","different nodes reuse the same evidence block"]];
 labels.forEach((a,i)=>{box(s,65+i*400,205,350,300,i===1?"#e8f4fb":"#f2f2f2",C.rule); text(s,a[0],90+i*400,235,300,62,27,{bold:true}); text(s,a[1],90+i*400,320,300,120,21,{color:C.muted});});
}
{
 const s=newSlide(p,4); title(s,"Existing methods regulate features globally; ACP-Gini is path-local","Research gap");
 const rows=[["VIF pruning","drops variables before training","information loss"],["RRF-style","fixed penalty for new features","not pairwise"],["Conditional importance","post-hoc diagnostic","does not change the tree"],["ACP-Gini","correlation with path ancestors","single-tree construction"]];
 rows.forEach((r,i)=>{const y=185+i*92; text(s,r[0],70,y,210,50,22,{bold:true,color:i===3?C.blue:C.ink}); text(s,r[1],310,y,420,50,21); text(s,r[2],790,y,390,50,21,{color:C.muted}); box(s,60,y+62,1130,1,C.rule,C.rule);});
}
{
 const s=newSlide(p,5); title(s,"Four questions separate redundancy, stability, accuracy, and size","Research questions");
 bullets(s,["RQ1 — Does bootstrap structure become more stable?","RQ2 — Are selected features less redundant and importance more reliable?","RQ3 — What accuracy–penalty tradeoff does α create?","RQ4 — Do trees become smaller under the same depth budget?"],90,185,1080,24,92);
}
{
 const s=newSlide(p,6); title(s,"Penalize correlation with distinct features already on the path","ACP-Gini idea");
 text(s,"Penalty(Xⱼ,T) = ∏ [1 − α |r(Xⱼ,Xₚ)|]",95,190,780,86,34,{bold:true}); text(s,"Xₚ ∈ ancestors(T), Xₚ ≠ Xⱼ",98,270,650,48,22,{color:C.muted});
 box(s,850,185,250,70,"#e8f4fb",C.blue); text(s,"root: no penalty",865,195,220,50,23,{bold:true,align:"center"}); box(s,850,310,250,70,"#ededed",C.rule); text(s,"child: compare to root",865,320,220,50,20,{align:"center"}); box(s,850,435,250,70,"#ededed",C.rule); text(s,"deeper: accumulate",865,445,220,50,20,{align:"center"});
 text(s,"No self-penalty • raw gain controls stopping • α=0 is exactly CART",95,520,940,52,23,{color:C.blue,bold:true});
}
{
 const s=newSlide(p,7); title(s,"The criterion changes ordering — never whether information exists","Formal behavior");
 const props=[["Backward compatible","α=0 gives CART"],["Monotone","higher |r| lowers score"],["Accumulating","several ancestors compound"],["No hard exclusion","positive factors for α<1"],["Repeat-safe","same feature is excluded from its own penalty"],["Raw-gain stopping","penalty cannot create an artificial leaf"]];
 props.forEach((a,i)=>{const col=i%3,row=Math.floor(i/3); box(s,65+col*395,185+row*195,350,155,"#f4f4f4",C.rule); text(s,a[0],85+col*395,205+row*195,310,42,23,{bold:true}); text(s,a[1],85+col*395,260+row*195,310,62,19,{color:C.muted});});
}
{
 const s=newSlide(p,8); title(s,"A conservative selection rule biases disagreements toward CART","Algorithm and selection");
 bullets(s,["Scan thresholds with cumulative class counts","Score each best raw gain × ancestor penalty","Tie-break by score, raw gain, then lowest feature index","Keep α values within 1 SD of CART macro-F1","Aggregate fold choices by lower quartile; RRF λ uses upper quartile"],80,175,690,20,72);
 box(s,850,190,290,250,C.ink,C.ink); text(s,"Cost",875,215,240,48,22,{color:C.cyan,bold:true}); text(s,"O(nd²)\none-time correlation\n+ path lookups",875,275,240,130,27,{color:C.white,align:"center"});
}
{
 const s=newSlide(p,9); title(s,"The benchmark isolates the split criterion under one complexity budget","Experimental setup");
 bullets(s,["5 real binary datasets + controlled synthetic groups","CART, VIF+CART, RRF-style, ACP-Gini","5-fold CV × 3 seeds = 15 held-out observations","B=50 bootstrap trees per training fold","max_depth=6; min_samples_leaf=5; paired Wilcoxon tests"],70,180,580,21,70);
 box(s,720,185,460,310,"#f3f3f3",C.rule); text(s,"Headline metrics",750,210,390,45,24,{bold:true}); text(s,"Accuracy / macro-F1 / AUC\n\nWeighted selected-set redundancy\n\nImportance rank correlation\nTop-5 Jaccard\nSplit-composition distance",750,270,390,210,20,{color:C.muted});
}
{
 const s=newSlide(p,10); title(s,"Synthetic groups recover broader signal coverage as α increases","Controlled evidence"); await image(s,"figures/fig2_synthetic.png",70,165,1140,470);
}
{
 const s=newSlide(p,11); title(s,"Accuracy stays flat while concentration improves — with a small noise cost","Alpha sensitivity"); await image(s,"figures/fig3_sensitivity.png",70,165,1140,470);
}
{
 const s=newSlide(p,12); title(s,"ACP-Gini lowers internal correlation on every collinear real dataset","Main result"); await image(s,"figures/fig4_redundancy.png",60,155,760,490); box(s,850,180,330,330,"#f3f3f3",C.rule); text(s,"What changes",880,205,270,44,24,{bold:true}); bullets(s,["Wine: strongest selected-setting reduction","WDBC: conservative α keeps accuracy unchanged","Sonar gap shrinks after lower-quartile refinement","Pima control remains practically flat"],875,270,270,18,62);
}
{
 const s=newSlide(p,13); title(s,"Stability is conditional: redundancy reduction can diversify rankings","Tradeoff"); await image(s,"figures/fig4b_importance_stability.png",60,155,740,480); text(s,"Wine",850,185,280,52,34,{bold:true,color:C.green}); text(s,"redundancy ↓\nstability ↑",850,240,280,80,25); text(s,"WDBC / Sonar",850,370,300,52,30,{bold:true,color:C.red}); text(s,"redundancy ↓\nrank stability ↓",850,425,280,80,25);
}
{
 const s=newSlide(p,14); title(s,"The path changes below an identical root","Qualitative comparison"); await image(s,"figures/fig5_tree_comparison.png",70,160,1140,470);
}
{
 const s=newSlide(p,15); title(s,"α is an operating control, not a universal optimum","Operating characteristics"); await image(s,"figures/fig6_real_alpha_sweep.png",70,155,820,480); box(s,920,180,280,300,"#f3f3f3",C.rule); text(s,"WDBC spot check",945,205,230,42,22,{bold:true}); text(s,"α=0.6\n46.5% less redundancy\naccuracy unchanged",945,270,230,125,25,{color:C.blue,bold:true,align:"center"}); text(s,"Three-seed average: 38.5% reduction with a 0.17 pp accuracy change",935,420,250,80,17,{color:C.muted,align:"center"});
}
{
 const s=newSlide(p,16); title(s,"ACP-Gini is a redundancy control with an honest stability tradeoff","Conclusion");
 bullets(s,["Primary: less internally correlated selected evidence","Synthetic: better group coverage and concentration","Predictive performance: statistically indistinguishable at selected α","Stability: improves on Wine/Pima, decreases on WDBC/Sonar","Next: categorical association, regression, ensembles, theory"],80,175,760,22,75);
 box(s,900,190,250,250,C.ink,C.ink); text(s,"Takeaway",930,220,190,42,20,{color:C.cyan,bold:true,align:"center"}); text(s,"Accuracy alone\ndoes not measure\nexplanation quality",925,285,200,135,27,{color:C.white,bold:true,align:"center"});
 text(s,"References: Breiman et al. (1984); Deng & Runger (2012, 2013); Strobl et al. (2008); Dormann et al. (2013); Rudin (2019).",80,620,1060,36,13,{color:C.muted});
}

await fs.mkdir(QA,{recursive:true});
for (const [i,s] of p.slides.items.entries()) {
  await writeBlob(path.join(QA,`slide-${String(i+1).padStart(2,"0")}.png`),await p.export({slide:s,format:"png",scale:1}));
  await fs.writeFile(path.join(QA,`slide-${String(i+1).padStart(2,"0")}.layout.json`),await (await s.export({format:"layout"})).text());
}
await writeBlob(path.join(QA,"montage.webp"),await p.export({format:"webp",montage:true,scale:1}));
const pptx=await PresentationFile.exportPptx(p); await pptx.save(OUT);
