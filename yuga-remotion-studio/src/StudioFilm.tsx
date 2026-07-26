import React from 'react';
import {AbsoluteFill, Easing, interpolate, Sequence, staticFile, useCurrentFrame} from 'remotion';
import {Audio} from '@remotion/media';
import {Book3D, BrassRule, DeepSpace, Grain, palette, Scene, TitleBlock} from './components/FilmPrimitives';
import {copy, Language} from './content';

const timings = {
  opening: {from:0,duration:150},
  map: {from:135,duration:165},
  timeline: {from:285,duration:195},
  gate: {from:465,duration:150},
  method: {from:600,duration:180},
  book: {from:765,duration:195},
  cta: {from:945,duration:135},
};

const Opening: React.FC<{lang:Language}> = ({lang}) => {
  const c=copy[lang];
  const frame=useCurrentFrame();
  const answer=interpolate(frame,[58,92],[0,1],{extrapolateLeft:'clamp',extrapolateRight:'clamp',easing:Easing.bezier(.16,1,.3,1)});
  return <Scene duration={timings.opening.duration} zoom={.04}>
    <DeepSpace horizon intensity={1.2}/>
    <div style={{position:'absolute',left:90,right:90,top:250}}>
      <div style={{fontFamily:'Arial,sans-serif',fontSize:22,letterSpacing:9,color:palette.brass,marginBottom:28}}>{c.openingTop}</div>
      <div style={{fontFamily:'Georgia,serif',fontSize:80,lineHeight:.98,color:palette.ivory,maxWidth:850}}>{c.openingMain}</div>
      <div style={{marginTop:28,opacity:answer,translate:`0 ${interpolate(answer,[0,1],[40,0])}px`,fontFamily:'Georgia,serif',fontSize:88,lineHeight:.95,color:palette.brass,textShadow:'0 0 55px rgba(233,173,66,.35)'}}>{c.openingAnswer}</div>
    </div>
    <div style={{position:'absolute',left:0,right:0,bottom:242,height:3,background:'linear-gradient(90deg, transparent 7%, rgba(233,173,66,.35), #ffd77c 50%, rgba(233,173,66,.35), transparent 93%)',scale:`${interpolate(frame,[0,125],[.1,1],{extrapolateRight:'clamp'})} 1`,boxShadow:'0 0 36px rgba(233,173,66,.7)'}}/>
    <div style={{position:'absolute',left:'50%',bottom:219,width:70,height:70,borderRadius:'50%',translate:'-50% 50%',background:'radial-gradient(circle,#fff8ca 0,#ffc04f 19%,rgba(233,173,66,.34) 42%,transparent 72%)',scale:interpolate(frame,[60,120],[.2,1.6],{extrapolateLeft:'clamp',extrapolateRight:'clamp'})}}/>
    <Grain/>
  </Scene>;
};

const MapScene: React.FC<{lang:Language}> = ({lang}) => {
  const c=copy[lang];
  const frame=useCurrentFrame();
  const draw=interpolate(frame,[10,125],[1700,0],{extrapolateLeft:'clamp',extrapolateRight:'clamp',easing:Easing.bezier(.16,1,.3,1)});
  const pulse=interpolate(frame%60,[0,30,60],[.35,1,.35]);
  return <Scene duration={timings.map.duration} zoom={.015}>
    <DeepSpace intensity={.65}/>
    <AbsoluteFill style={{background:'linear-gradient(180deg,rgba(3,8,18,.25),rgba(2,7,15,.86))'}}/>
    <svg width="1080" height="1920" viewBox="0 0 1080 1920" style={{position:'absolute',inset:0,opacity:.82}}>
      <defs>
        <linearGradient id="line" x1="0" y1="0" x2="1" y2="1"><stop stopColor="#e9ad42"/><stop offset="1" stopColor="#466f91"/></linearGradient>
        <filter id="glow"><feGaussianBlur stdDeviation="5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
      </defs>
      {[0,1,2,3,4,5,6].map((i)=><path key={i} d={`M ${-80+i*28} ${1180-i*74} C ${180+i*52} ${860-i*34}, ${450-i*20} ${1280-i*64}, ${650+i*32} ${960-i*58} S ${990-i*20} ${710+i*50}, ${1160} ${850-i*27}`} fill="none" stroke={i===3?'url(#line)':'rgba(132,165,184,.24)'} strokeWidth={i===3?3:1.3} strokeDasharray="12 10" strokeDashoffset={draw+i*38} filter={i===3?'url(#glow)':undefined}/>) }
      <path d="M140 1320 C270 1050 320 920 475 815 C580 745 745 770 900 590" fill="none" stroke="#f0c56f" strokeWidth="3" strokeDasharray="1700" strokeDashoffset={draw} filter="url(#glow)"/>
      {[{x:140,y:1320},{x:475,y:815},{x:900,y:590}].map((p,i)=><g key={i}><circle cx={p.x} cy={p.y} r={13+9*pulse} fill="rgba(233,173,66,.12)"/><circle cx={p.x} cy={p.y} r="5" fill="#ffd985"/></g>)}
      <g opacity=".28" stroke="#7792a8" strokeWidth="1">{Array.from({length:9}).map((_,i)=><line key={'v'+i} x1={90+i*115} x2={90+i*115} y1="420" y2="1520"/>)}{Array.from({length:10}).map((_,i)=><line key={'h'+i} x1="70" x2="1010" y1={470+i*105} y2={470+i*105}/>)}</g>
    </svg>
    <div style={{position:'absolute',left:80,right:80,top:180}}><TitleBlock kicker="THE LUCID CARTOGRAPHER" title={c.mapTitle} subtitle={c.mapBody}/></div>
    <div style={{position:'absolute',left:80,bottom:170}}><BrassRule width={620}/></div>
    <Grain/>
  </Scene>;
};

const TimelineScene: React.FC<{lang:Language}> = ({lang}) => {
  const c=copy[lang];
  const frame=useCurrentFrame();
  return <Scene duration={timings.timeline.duration} zoom={.01}>
    <DeepSpace intensity={.5}/>
    <div style={{position:'absolute',left:70,right:70,top:135}}><TitleBlock title={c.timelineTitle} maxWidth={950}/></div>
    <div style={{position:'absolute',left:70,right:70,top:530,perspective:1400}}>
      {c.tracks.map((track,i)=>{
        const p=interpolate(frame,[18+i*17,90+i*17],[0,1],{extrapolateLeft:'clamp',extrapolateRight:'clamp',easing:Easing.bezier(.16,1,.3,1)});
        const active=i===2;
        return <div key={track[0]} style={{position:'relative',height:220,opacity:p,translate:`${interpolate(p,[0,1],[150,0])}px 0`,transform:`rotateX(${7-i*2}deg)`}}>
          <div style={{fontFamily:'Arial,sans-serif',fontSize:21,letterSpacing:4.5,color:active?palette.brass:'rgba(240,229,207,.58)',marginBottom:23}}>{track[0]}</div>
          <div style={{height:3,background:active?'linear-gradient(90deg,#8b5d1d,#ffe6a1,#8b5d1d)':'linear-gradient(90deg,rgba(85,112,136,.2),rgba(144,171,192,.72),rgba(85,112,136,.2))',scale:`${p} 1`,transformOrigin:'left center',boxShadow:active?'0 0 30px rgba(233,173,66,.55)':'0 0 18px rgba(80,130,180,.2)'}}/>
          <div style={{display:'flex',justifyContent:'flex-end',marginTop:20,fontFamily:'Georgia,serif',fontSize:active?66:48,color:active?palette.ivory:'rgba(240,229,207,.56)',textShadow:active?'0 0 36px rgba(233,173,66,.38)':'none'}}>{track[1]}</div>
          {active?<div style={{position:'absolute',right:4,top:34,width:18,height:18,borderRadius:'50%',background:'#ffe4a1',boxShadow:'0 0 40px #e9ad42'}}/>:null}
        </div>;
      })}
    </div>
    <Grain/>
  </Scene>;
};

const GateScene: React.FC<{lang:Language}> = ({lang}) => {
  const c=copy[lang];
  const frame=useCurrentFrame();
  const open=interpolate(frame,[15,92],[0,1],{extrapolateRight:'clamp',easing:Easing.bezier(.16,1,.3,1)});
  return <Scene duration={timings.gate.duration} zoom={.035}>
    <DeepSpace horizon intensity={1.4}/>
    <div style={{position:'absolute',left:'50%',top:'50%',width:730,height:730,borderRadius:'50%',translate:'-50% -50%',border:`${interpolate(open,[0,1],[28,3])}px solid rgba(233,173,66,${.12+.64*open})`,boxShadow:`0 0 ${120*open}px rgba(233,173,66,.5), inset 0 0 ${100*open}px rgba(233,173,66,.18)`,scale:interpolate(open,[0,1],[.35,1])}}/>
    <div style={{position:'absolute',left:70,right:70,top:680,textAlign:'center',opacity:open,scale:interpolate(open,[0,1],[.8,1])}}>
      <div style={{fontFamily:'Georgia,serif',fontSize:108,letterSpacing:5,color:palette.ivory,textShadow:'0 0 55px rgba(233,173,66,.5)'}}>{c.gateTitle}</div>
      <div style={{fontFamily:'Arial,sans-serif',fontSize:24,letterSpacing:9,color:palette.brass,marginTop:24}}>{c.gateSub}</div>
    </div>
    <Grain/>
  </Scene>;
};

const MethodScene: React.FC<{lang:Language}> = ({lang}) => {
  const c=copy[lang];
  const frame=useCurrentFrame();
  return <Scene duration={timings.method.duration} zoom={.012}>
    <DeepSpace intensity={.4}/>
    <div style={{position:'absolute',left:75,right:75,top:140}}><TitleBlock title={c.methodTitle} subtitle={c.methodSub}/></div>
    <div style={{position:'absolute',left:70,right:70,top:720,display:'grid',gridTemplateColumns:'1fr 1fr',gap:22}}>
      {c.methodLabels.map((label,i)=>{
        const p=interpolate(frame,[28+i*14,82+i*14],[0,1],{extrapolateLeft:'clamp',extrapolateRight:'clamp',easing:Easing.bezier(.16,1,.3,1)});
        const colors=[palette.brass,'#6f9a8d','#7895b3',palette.rust];
        return <div key={label} style={{height:250,position:'relative',overflow:'hidden',border:'1px solid rgba(240,229,207,.15)',background:'linear-gradient(145deg,rgba(255,255,255,.055),rgba(255,255,255,.012))',opacity:p,translate:`0 ${interpolate(p,[0,1],[55,0])}px`,boxShadow:'0 26px 70px rgba(0,0,0,.28)'}}>
          <div style={{position:'absolute',inset:0,backgroundImage:`url(${staticFile('assets/paper.svg')})`,backgroundSize:'cover',opacity:.055,mixBlendMode:'screen'}}/>
          <div style={{position:'absolute',left:28,top:28,width:42,height:4,background:colors[i],boxShadow:`0 0 22px ${colors[i]}`}}/>
          <div style={{position:'absolute',left:28,right:24,bottom:32,fontFamily:'Arial,sans-serif',fontSize:23,letterSpacing:3.2,color:palette.ivory}}>{label}</div>
          <svg width="100%" height="100%" viewBox="0 0 450 250" style={{position:'absolute',inset:0,opacity:.22}}><path d={`M -20 ${170-i*18} C 90 ${75+i*10}, 180 ${230-i*12}, 300 ${112+i*8} S 430 ${190-i*22}, 480 ${80+i*25}`} fill="none" stroke={colors[i]} strokeWidth="2" strokeDasharray="8 8" strokeDashoffset={interpolate(frame,[0,180],[220,0])}/></svg>
        </div>;
      })}
    </div>
    <Grain/>
  </Scene>;
};

const BookScene: React.FC<{lang:Language}> = ({lang}) => {
  const c=copy[lang];
  const frame=useCurrentFrame();
  const light=interpolate(frame,[20,100,190],[0,.8,.2],{extrapolateLeft:'clamp',extrapolateRight:'clamp'});
  return <Scene duration={timings.book.duration} zoom={.02}>
    <DeepSpace horizon intensity={1.1}/>
    <div style={{position:'absolute',left:'50%',top:900,width:830,height:500,borderRadius:'50%',translate:'-50% -50%',background:`radial-gradient(ellipse,rgba(233,173,66,${.4*light}),transparent 68%)`,filter:'blur(14px)'}}/>
    <div style={{position:'absolute',left:70,right:70,top:110,textAlign:'center'}}><div style={{fontFamily:'Arial,sans-serif',fontSize:20,letterSpacing:6,color:palette.brass}}>{c.bookKicker}</div></div>
    <div style={{position:'absolute',left:'50%',top:410,translate:'-50% 0'}}><Book3D cover={lang}/></div>
    <div style={{position:'absolute',left:70,right:70,bottom:112,textAlign:'center',opacity:interpolate(frame,[80,145],[0,1],{extrapolateLeft:'clamp',extrapolateRight:'clamp'})}}>
      <div style={{fontFamily:'Georgia,serif',fontSize:48,color:palette.ivory}}>{c.bookTitle}</div>
      <div style={{fontFamily:'Arial,sans-serif',fontSize:22,letterSpacing:1.5,color:'rgba(240,229,207,.68)',marginTop:15}}>{c.bookSub}</div>
    </div>
    <Grain/>
  </Scene>;
};

const CtaScene: React.FC<{lang:Language}> = ({lang}) => {
  const c=copy[lang];
  const frame=useCurrentFrame();
  const p=interpolate(frame,[8,56],[0,1],{extrapolateRight:'clamp',easing:Easing.bezier(.16,1,.3,1)});
  return <Scene duration={timings.cta.duration} lead={8} trail={20} zoom={.015}>
    <DeepSpace horizon intensity={1.3}/>
    <div style={{position:'absolute',inset:0,background:'radial-gradient(circle at 50% 60%,rgba(233,173,66,.18),transparent 28%)'}}/>
    <div style={{position:'absolute',left:72,right:72,top:510,textAlign:'center',opacity:p,translate:`0 ${interpolate(p,[0,1],[50,0])}px`}}>
      <div style={{display:'flex',justifyContent:'center',marginBottom:45}}><BrassRule width={620}/></div>
      <div style={{fontFamily:'Georgia,serif',fontSize:70,lineHeight:1.05,color:palette.ivory}}>{c.cta}</div>
      <div style={{fontFamily:'Arial,sans-serif',fontSize:21,letterSpacing:7,color:palette.brass,marginTop:58}}>{c.by}</div>
    </div>
    <div style={{position:'absolute',left:'50%',bottom:260,width:22,height:22,borderRadius:'50%',translate:'-50% 0',background:'#ffe7a6',boxShadow:'0 0 60px 22px rgba(233,173,66,.42)',scale:interpolate(frame,[0,110],[.2,1.4],{extrapolateRight:'clamp'})}}/>
    <Grain/>
  </Scene>;
};

export const StudioFilm: React.FC<{lang:Language}> = ({lang}) => <AbsoluteFill style={{backgroundColor:palette.night}}>
  <Audio src={staticFile('audio/yuga-studio-master.wav')} volume={1}/>
  <Sequence from={timings.opening.from} durationInFrames={timings.opening.duration}><Opening lang={lang}/></Sequence>
  <Sequence from={timings.map.from} durationInFrames={timings.map.duration}><MapScene lang={lang}/></Sequence>
  <Sequence from={timings.timeline.from} durationInFrames={timings.timeline.duration}><TimelineScene lang={lang}/></Sequence>
  <Sequence from={timings.gate.from} durationInFrames={timings.gate.duration}><GateScene lang={lang}/></Sequence>
  <Sequence from={timings.method.from} durationInFrames={timings.method.duration}><MethodScene lang={lang}/></Sequence>
  <Sequence from={timings.book.from} durationInFrames={timings.book.duration}><BookScene lang={lang}/></Sequence>
  <Sequence from={timings.cta.from} durationInFrames={timings.cta.duration}><CtaScene lang={lang}/></Sequence>
</AbsoluteFill>;
