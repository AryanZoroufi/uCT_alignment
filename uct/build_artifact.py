"""Combine the exported 7-DOF scene JSON with the WebGL viewer template into a
single self-contained HTML artifact."""
import json
from pathlib import Path

SCRATCH = Path("/tmp/claude-1000/-home-aryan-Projects-uct-backup-uCT-alignment/"
               "2243f74f-59e2-4926-a697-e6fdd62c17ea/scratchpad")
scene = (SCRATCH / "align7dof_scene.json").read_text()

HTML = r"""<style>
  :root{
    --bg:#0a0d12; --panel:rgba(18,23,31,.86); --line:rgba(255,255,255,.09);
    --text:#dbe2ec; --muted:#8b95a4; --accent:#46c6d8; --accent-dim:#1c3a42;
    --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
    --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,system-ui,sans-serif;
  }
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--text);
    font-family:var(--sans);-webkit-font-smoothing:antialiased;overflow:hidden;
    overscroll-behavior:none;touch-action:none;}
  #gl{position:fixed;inset:0;width:100%;height:100%;display:block;cursor:grab;}
  #gl:active{cursor:grabbing;}
  .head{position:fixed;top:0;left:0;padding:14px 16px;pointer-events:none;z-index:5;}
  .head h1{margin:0;font-size:13px;font-weight:600;letter-spacing:.14em;
    text-transform:uppercase;color:var(--text);}
  .head p{margin:3px 0 0;font-size:11.5px;color:var(--muted);max-width:60ch;
    font-family:var(--mono);letter-spacing:.01em;}
  .head b{color:var(--accent);font-weight:600;}

  .deck{position:fixed;left:0;right:0;bottom:0;z-index:6;
    background:var(--panel);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
    border-top:1px solid var(--line);padding:12px 14px calc(12px + env(safe-area-inset-bottom));
    display:flex;flex-direction:column;gap:11px;transition:transform .28s ease;}
  .deck.collapsed{transform:translateY(calc(100% - 42px));}
  .deckbar{display:flex;align-items:center;justify-content:space-between;gap:10px;}
  .deckbar .lbl{font-size:10.5px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);}
  .caret{appearance:none;background:none;border:1px solid var(--line);color:var(--muted);
    border-radius:7px;width:30px;height:26px;font-size:13px;cursor:pointer;}
  .row{display:flex;flex-wrap:wrap;gap:7px;}
  .chip{appearance:none;font-family:var(--mono);font-size:12px;color:var(--muted);
    background:rgba(255,255,255,.03);border:1px solid var(--line);border-radius:8px;
    padding:7px 11px;cursor:pointer;display:flex;align-items:center;gap:7px;
    letter-spacing:.02em;transition:all .15s;}
  .chip .dot{width:8px;height:8px;border-radius:50%;background:currentColor;opacity:.5;}
  .chip[aria-pressed="true"]{color:var(--text);background:var(--accent-dim);
    border-color:var(--accent);}
  .chip[aria-pressed="true"] .dot{opacity:1;background:var(--accent);
    box-shadow:0 0 8px var(--accent);}
  .split{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start;}
  .grp{display:flex;flex-direction:column;gap:7px;min-width:0;}
  .grp>.lbl{font-size:10px;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);}
  .legend{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:11px;color:var(--muted);
    font-family:var(--mono);}
  .legend span{display:flex;align-items:center;gap:6px;}
  .sw{width:11px;height:11px;border-radius:3px;flex:none;}
  .btn{appearance:none;font-family:var(--mono);font-size:11.5px;color:var(--muted);
    background:none;border:1px solid var(--line);border-radius:8px;padding:7px 11px;cursor:pointer;}
  .btn:hover{color:var(--text);border-color:var(--accent);}
  .hint{font-size:10.5px;color:var(--muted);font-family:var(--mono);letter-spacing:.02em;}
  button:focus-visible,.chip:focus-visible{outline:2px solid var(--accent);outline-offset:2px;}
  @media(min-width:720px){
    .deck{left:auto;right:14px;bottom:14px;width:330px;border:1px solid var(--line);
      border-radius:14px;}
    .deck.collapsed{transform:translateY(calc(100% - 42px));}
  }
</style>

<canvas id="gl"></canvas>

<div class="head">
  <h1>µCT · Tibia Segmentation</h1>
  <p>The detailed tibia surface, split into atlas segments — each piece a separate
  mesh. Tibia = <b>red 1</b> &amp; <b>blue 4</b>. Drag to rotate · pinch to zoom.</p>
</div>

<div class="deck" id="deck">
  <div class="deckbar">
    <span class="lbl">Specimens · Layers</span>
    <button class="caret" id="caret" aria-label="collapse panel">▾</button>
  </div>
  <div class="row" id="specimens"></div>
  <div class="grp">
    <span class="lbl">Segments · tap to toggle</span>
    <div class="row" id="parts"></div>
  </div>
  <div class="deckbar">
    <span class="hint" id="hint"></span>
    <button class="btn" id="reset">Reset view</button>
  </div>
</div>

<script type="application/json" id="scene">__SCENE__</script>
<script>
(function(){
"use strict";
const scene = JSON.parse(document.getElementById("scene").textContent);
const LCOL = {1:[230,25,75],2:[60,180,75],3:[255,215,20],4:[0,130,200],
  5:[245,130,48],6:[145,30,180],7:[70,240,240],8:[240,50,230],9:[170,110,40]};

const cv = document.getElementById("gl");
const gl = cv.getContext("webgl2",{antialias:true,alpha:false});
if(!gl){document.body.innerHTML="<p style='color:#dbe2ec;font-family:sans-serif;padding:24px'>WebGL2 not supported on this device.</p>";return;}

function b64(s,T){const b=atob(s),n=b.length,u=new Uint8Array(n);for(let i=0;i<n;i++)u[i]=b.charCodeAt(i);return new T(u.buffer);}

const V=`#version 300 es
in vec3 aPos; in vec3 aCol;
uniform mat4 uMVP;
out vec3 vC; out vec3 vP;
void main(){ vC=aCol; vP=aPos; gl_Position=uMVP*vec4(aPos,1.0); }`;
const F=`#version 300 es
precision highp float;
in vec3 vC; in vec3 vP;
uniform vec3 uEye;
out vec4 o;
void main(){
  vec3 N=normalize(cross(dFdx(vP),dFdy(vP)));   // flat normal, no normal buffer
  vec3 L=normalize(uEye-vP);
  float d=abs(dot(N,L));
  o=vec4(vC*(0.32+0.68*d),1.0);
}`;
function sh(t,s){const x=gl.createShader(t);gl.shaderSource(x,s);gl.compileShader(x);
  if(!gl.getShaderParameter(x,gl.COMPILE_STATUS))console.error(gl.getShaderInfoLog(x));return x;}
const prog=gl.createProgram();
gl.attachShader(prog,sh(gl.VERTEX_SHADER,V));gl.attachShader(prog,sh(gl.FRAGMENT_SHADER,F));
gl.bindAttribLocation(prog,0,"aPos");gl.bindAttribLocation(prog,1,"aCol");
gl.linkProgram(prog);gl.useProgram(prog);
const uMVP=gl.getUniformLocation(prog,"uMVP"), uEye=gl.getUniformLocation(prog,"uEye");

let lo=[1e9,1e9,1e9], hi=[-1e9,-1e9,-1e9];
const meshes=scene.meshes.map(m=>{
  const pos=b64(m.pos,Float32Array), col=b64(m.col,Uint8Array), idx=b64(m.idx,Uint16Array);
  for(let i=0;i<pos.length;i+=3){for(let k=0;k<3;k++){const v=pos[i+k];
    if(v<lo[k])lo[k]=v; if(v>hi[k])hi[k]=v;}}
  const vao=gl.createVertexArray(); gl.bindVertexArray(vao);
  const pb=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,pb);
  gl.bufferData(gl.ARRAY_BUFFER,pos,gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);
  const cb=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,cb);
  gl.bufferData(gl.ARRAY_BUFFER,col,gl.STATIC_DRAW);
  gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1,3,gl.UNSIGNED_BYTE,true,0,0);
  const eb=gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,eb);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,idx,gl.STATIC_DRAW);
  gl.bindVertexArray(null);
  return {vao,count:idx.length,group:m.group,label:m.label};
});
const ctr=[(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,(lo[2]+hi[2])/2];
const rad=Math.max(hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2])*0.5||10;

// ---- mat4 ----
function persp(f,a,n,fa){const t=1/Math.tan(f/2);return[t/a,0,0,0, 0,t,0,0, 0,0,(fa+n)/(n-fa),-1, 0,0,2*fa*n/(n-fa),0];}
function look(e,c,u){let z=[e[0]-c[0],e[1]-c[1],e[2]-c[2]];let zl=Math.hypot(z[0],z[1],z[2]);z=[z[0]/zl,z[1]/zl,z[2]/zl];
  let x=[u[1]*z[2]-u[2]*z[1],u[2]*z[0]-u[0]*z[2],u[0]*z[1]-u[1]*z[0]];let xl=Math.hypot(x[0],x[1],x[2])||1;x=[x[0]/xl,x[1]/xl,x[2]/xl];
  let y=[z[1]*x[2]-z[2]*x[1],z[2]*x[0]-z[0]*x[2],z[0]*x[1]-z[1]*x[0]];
  return[x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, -(x[0]*e[0]+x[1]*e[1]+x[2]*e[2]),-(y[0]*e[0]+y[1]*e[1]+y[2]*e[2]),-(z[0]*e[0]+z[1]*e[1]+z[2]*e[2]),1];}
function mul(a,b){const o=new Array(16);for(let r=0;r<4;r++)for(let c=0;c<4;c++){let s=0;for(let k=0;k<4;k++)s+=a[k*4+r]*b[c*4+k];o[c*4+r]=s;}return o;}

// ---- camera ----
const AZ0=0.7,EL0=0.32,D0=rad*3.0;
let az=AZ0,el=EL0,dist=D0;
const groupsOn=scene.groups.map(()=>true);
const labelsOn={};
function eye(){const ce=Math.cos(el),se=Math.sin(el);return[ctr[0]+dist*ce*Math.sin(az),ctr[1]+dist*se,ctr[2]+dist*ce*Math.cos(az)];}
function resize(){const dpr=Math.min(devicePixelRatio||1,2);cv.width=innerWidth*dpr;cv.height=innerHeight*dpr;gl.viewport(0,0,cv.width,cv.height);draw();}
function draw(){
  gl.clearColor(0.039,0.051,0.071,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);
  const P=persp(0.85,cv.width/cv.height,rad*0.05,rad*40);
  const e=eye(); const mvp=mul(P,look(e,ctr,[0,1,0]));
  gl.uniformMatrix4fv(uMVP,false,new Float32Array(mvp)); gl.uniform3fv(uEye,e);
  for(const m of meshes){ if(!groupsOn[m.group])continue; if(!labelsOn[m.label])continue;
    gl.bindVertexArray(m.vao); gl.drawElements(gl.TRIANGLES,m.count,gl.UNSIGNED_SHORT,0);}
}

// ---- input (pointer + pinch) ----
const pts=new Map(); let pinch0=0,dist0=0;
cv.addEventListener("pointerdown",e=>{cv.setPointerCapture(e.pointerId);pts.set(e.pointerId,[e.clientX,e.clientY]);
  if(pts.size===2){const a=[...pts.values()];pinch0=Math.hypot(a[0][0]-a[1][0],a[0][1]-a[1][1]);dist0=dist;}});
cv.addEventListener("pointermove",e=>{if(!pts.has(e.pointerId))return;const p=pts.get(e.pointerId);
  if(pts.size===1){az-=(e.clientX-p[0])*0.006; el-=(e.clientY-p[1])*0.006; el=Math.max(-1.5,Math.min(1.5,el));}
  pts.set(e.pointerId,[e.clientX,e.clientY]);
  if(pts.size===2){const a=[...pts.values()];const d=Math.hypot(a[0][0]-a[1][0],a[0][1]-a[1][1]);
    if(pinch0>0){dist=Math.max(rad*0.6,Math.min(rad*12,dist0*pinch0/d));}}
  draw();});
function up(e){pts.delete(e.pointerId);if(pts.size<2)pinch0=0;}
cv.addEventListener("pointerup",up);cv.addEventListener("pointercancel",up);
cv.addEventListener("wheel",e=>{e.preventDefault();dist=Math.max(rad*0.6,Math.min(rad*12,dist*Math.exp(e.deltaY*0.0012)));draw();},{passive:false});

// ---- UI ----
const specEl=document.getElementById("specimens");
scene.groups.forEach((g,i)=>{const b=document.createElement("button");b.className="chip";
  b.setAttribute("aria-pressed","true");b.innerHTML='<span class="dot"></span>'+g;
  b.onclick=()=>{groupsOn[i]=!groupsOn[i];b.setAttribute("aria-pressed",groupsOn[i]);draw();};
  specEl.appendChild(b);});
const labels=[...new Set(meshes.map(m=>m.label))].sort((a,b)=>a-b);
labels.forEach(L=>{labelsOn[L]=true;});
function lname(L){return L===1?"Tibia · part 1":L===4?"Tibia · part 4":"bone "+L;}
const partsEl=document.getElementById("parts");
labels.forEach(L=>{const c=LCOL[L]||[200,200,200];
  const b=document.createElement("button");b.className="chip";b.setAttribute("aria-pressed","true");
  b.innerHTML='<span class="dot" style="background:rgb('+c.join(",")+');box-shadow:none;opacity:1"></span>'+lname(L);
  b.onclick=()=>{labelsOn[L]=!labelsOn[L];b.setAttribute("aria-pressed",labelsOn[L]);draw();};
  partsEl.appendChild(b);});
document.getElementById("reset").onclick=()=>{az=AZ0;el=EL0;dist=D0;draw();};
const deck=document.getElementById("deck");
document.getElementById("caret").onclick=e=>{deck.classList.toggle("collapsed");
  e.currentTarget.textContent=deck.classList.contains("collapsed")?"▴":"▾";};
document.getElementById("hint").textContent=scene.groups.length+" specimens · "+
  scene.meshes.reduce((a,m)=>a+m.nfaces,0).toLocaleString()+" faces";

addEventListener("resize",resize); resize();
})();
</script>"""

out = SCRATCH / "align7dof_viewer.html"
out.write_text(HTML.replace("__SCENE__", scene))
print(f"wrote {out}  ({out.stat().st_size/1e6:.2f} MB)")
