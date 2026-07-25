import React from 'react';
import {registerRoot, Composition, Still, AbsoluteFill, Img, Sequence, Easing, interpolate, staticFile, useCurrentFrame} from 'remotion';
import {Audio} from '@remotion/media';

type Locale = 'fr' | 'en';
type FilmProps = {locale: Locale; cover: string};

const P = {
  night:'#03080d', midnight:'#07131b', slate:'#102832', paper:'#eee5d4', mist:'#b9c2c0',
  brass:'#c18d45', gold:'#efbb63', ox:'#62817a', signal:'#98564d', white:'#faf7ef'
};
const SERIF='"EB Garamond", Georgia, serif';
const SANS='Inter, "Helvetica Neue", Arial, sans-serif';
const ease=Easing.bezier(0.16,1,0.3,1);
const clamp={extrapolateLeft:'clamp' as const,extrapolateRight:'clamp' as const};
const fade=(f:number,d:number)=>interpolate(f,[0,18,d-18,d],[0,1,1,0],{...clamp,easing:ease});

const C={
fr:{
  coast:'RIVE SUD DE LA MÉDITERRANÉE · 2026', a:'Et si notre époque n’était pas un chaos…', b:'mais un seuil ?',
  maps:'TROIS CARTES · TROIS ÉCHELLES', conflict:'Elles ne racontent pas la même histoire.',
  systems:[['SYSTÈME PURANIQUE','+400 000 ANS'],['RÉFORME DE YUKTESWAR','4099'],['ENSEIGNEMENT CONTEMPORAIN','2044–2082']],
  footer:'LE LIVRE PLACE LES TROIS SYSTÈMES CÔTE À CÔTE — ET MONTRE LEURS CALCULS.',
  calc:'ARITHMÉTIQUE · SOURCES · CORRECTIONS', evidence:'Une chronologie ne vaut pas par son aura.', evidence2:'Elle vaut par ce qu’elle permet de vérifier.',
  wheel:'LE CYCLE COMME CARTE', wheelLead:'Quatre âges. Deux mouvements. Une lecture qui reste discutable.',
  method:'QUATRE REGISTRES SUR CHAQUE PAGE', map:'La carte reste une carte —', fog:'dessinée dans la brume.',
  cats:[['FAIT','vérifié et sourcé'],['TRADITION','rapportée sans être imposée'],['INTERPRÉTATION','raisonnée et signalée'],['CONJECTURE','explorée sans être déguisée']],
  no:'NI PROPHÉTIE.\nNI DÉMOLITION.', marsh:'Les marais ne sont pas effacés. Ils sont marqués.',
  lucid:'UNE CARTE LUCIDE DU CYCLE DES YUGAS', cta:'DÉCOUVREZ LA CARTE COMPLÈTE', author:'THE LUCID CARTOGRAPHER'
},
en:{
  coast:'SOUTHERN MEDITERRANEAN SHORE · 2026', a:'What if our era were not a collapse…', b:'but a threshold?',
  maps:'THREE MAPS · THREE SCALES', conflict:'They do not tell the same story.',
  systems:[['PURANIC SYSTEM','400,000+ YEARS'],['YUKTESWAR REFORM','4099'],['CONTEMPORARY TEACHING','2044–2082']],
  footer:'THE BOOK SETS THE THREE SYSTEMS SIDE BY SIDE — AND SHOWS THEIR ARITHMETIC.',
  calc:'ARITHMETIC · SOURCES · CORRECTIONS', evidence:'A chronology is not credible because it feels ancient.', evidence2:'It is credible to the extent that it can be checked.',
  wheel:'THE CYCLE AS A MAP', wheelLead:'Four ages. Two movements. A reading that remains debatable.',
  method:'FOUR REGISTERS ON EVERY PAGE', map:'A map remains a map —', fog:'drawn through the fog.',
  cats:[['FACT','verified and sourced'],['TRADITION','reported without imposing it'],['INTERPRETATION','reasoned and identified'],['CONJECTURE','explored without disguise']],
  no:'NEITHER PROPHECY\nNOR DEMOLITION', marsh:'The marshes are not erased. They are marked.',
  lucid:'A LUCID MAP OF THE YUGA CYCLE', cta:'DISCOVER THE COMPLETE MAP', author:'THE LUCID CARTOGRAPHER'
}} as const;

const Texture:React.FC=()=>{
  const f=useCurrentFrame();
  return <AbsoluteFill style={{pointerEvents:'none',mixBlendMode:'screen',opacity:.055}}>
    <svg width="100%" height="100%" viewBox="0 0 1080 1920"><filter id="n"><feTurbulence type="fractalNoise" baseFrequency=".82" numOctaves="2" seed={(f*37)%181}/><feColorMatrix type="saturate" values="0"/></filter><rect width="1080" height="1920" filter="url(#n)" opacity=".55"/></svg>
  </AbsoluteFill>;
};
const Grid:React.FC<{o?:number}>=({o=.12})=>{
 const f=useCurrentFrame(); const y=interpolate(f,[0,1080],[0,-25]);
 return <AbsoluteFill style={{opacity:o,translate:`0 ${y}px`}}><svg width="100%" height="100%" viewBox="0 0 1080 1920"><defs><pattern id="m" width="48" height="48" patternUnits="userSpaceOnUse"><path d="M48 0H0V48" fill="none" stroke={P.ox} strokeWidth=".7" opacity=".34"/></pattern><pattern id="M" width="240" height="240" patternUnits="userSpaceOnUse"><rect width="240" height="240" fill="url(#m)"/><path d="M240 0H0V240" fill="none" stroke={P.ox} strokeWidth="1.1" opacity=".5"/></pattern><radialGradient id="gf"><stop offset="0" stopColor="white"/><stop offset="1" stopColor="white" stopOpacity="0"/></radialGradient><mask id="gm"><rect width="1080" height="1920" fill="url(#gf)"/></mask></defs><rect width="1080" height="1920" fill="url(#M)" mask="url(#gm)"/></svg></AbsoluteFill>;
};
const Base:React.FC<{cover:string;children:React.ReactNode;grid?:number}>=({cover,children,grid=.12})=><AbsoluteFill style={{background:`radial-gradient(circle at 58% 23%,${P.slate},${P.midnight} 40%,${P.night} 76%,#010305)`,color:P.paper,overflow:'hidden'}}>
  <Img src={staticFile(cover)} style={{position:'absolute',inset:-120,width:1320,height:2160,objectFit:'cover',filter:'blur(44px) brightness(.24) saturate(.78)',opacity:.62}}/>
  <AbsoluteFill style={{background:'linear-gradient(180deg,rgba(1,5,9,.32),rgba(1,5,9,.82))'}}/><Grid o={grid}/>{children}<Texture/>
</AbsoluteFill>;
const Label:React.FC<{children:React.ReactNode}>=({children})=><div style={{fontFamily:SANS,fontSize:19,letterSpacing:4,color:P.brass}}>{children}</div>;

const Coast:React.FC<FilmProps & {d:number}>=({locale,cover,d})=>{const f=useCurrentFrame(),c=C[locale],p=interpolate(f,[6,96],[0,1],{...clamp,easing:ease}),push=interpolate(f,[0,d],[1.02,1.10]);return <Base cover={cover} grid={.04}>
 <div style={{position:'absolute',inset:0,scale:push,background:'linear-gradient(180deg,rgba(3,14,22,.3),rgba(1,4,7,.82))'}}/>
 <svg width="1080" height="1920" style={{position:'absolute',inset:0}}><defs><linearGradient id="h" x1="0" x2="1"><stop stopColor={P.gold} stopOpacity="0"/><stop offset=".48" stopColor={P.gold} stopOpacity=".9"/><stop offset="1" stopColor={P.gold} stopOpacity="0"/></linearGradient><linearGradient id="sea" y2="1"><stop stopColor={P.ox} stopOpacity=".18"/><stop offset="1" stopColor="#010306" stopOpacity=".98"/></linearGradient></defs><rect y="1100" width="1080" height="820" fill="url(#sea)"/><rect x="75" y="1095" width={930*p} height="3" fill="url(#h)"/><path d="M0 1550L150 1410l120 88 120-145 150 165 155-72 145 105 140-94 100 45v428H0Z" fill="#010306"/><path d="M-120 1320A660 480 0 0 1 1200 1320" fill="none" stroke={P.gold} strokeWidth="3" opacity={.72*p}/></svg>
 <div style={{position:'absolute',left:92,top:104}}><Label>{c.coast}</Label></div>
 <div style={{position:'absolute',left:92,right:90,top:420,opacity:fade(f,d)}}><div style={{fontFamily:SERIF,fontSize:locale==='fr'?84:78,lineHeight:1.02,maxWidth:840}}>{c.a}</div><div style={{fontFamily:SERIF,fontStyle:'italic',fontSize:106,lineHeight:1,color:P.gold,marginTop:32}}>{c.b}</div></div>
 <div style={{position:'absolute',left:92,bottom:88,fontFamily:SANS,fontSize:17,letterSpacing:3,color:P.mist,opacity:.7}}>36.7538° N · 3.0588° E</div>
 </Base>};

const Chronologies:React.FC<FilmProps & {d:number}>=({locale,cover,d})=>{const f=useCurrentFrame(),c=C[locale];return <Base cover={cover}>
 <div style={{position:'absolute',left:96,top:112}}><Label>{c.maps}</Label></div><div style={{position:'absolute',left:96,right:96,top:250,fontFamily:SERIF,fontSize:76,lineHeight:1.02,opacity:fade(f,d)}}>{c.conflict}</div>
 <div style={{position:'absolute',left:96,right:96,top:610}}>{c.systems.map((s,i)=>{const p=interpolate(f,[12+i*16,54+i*16],[0,1],{...clamp,easing:ease});const col=[P.mist,P.ox,P.gold][i];return <div key={s[0]} style={{position:'relative',height:270,opacity:p,translate:`${interpolate(p,[0,1],[-70*i,0])}px 0`}}><div style={{fontFamily:SANS,fontSize:19,letterSpacing:3,color:col}}>{s[0]}</div><div style={{position:'absolute',right:0,top:-10,fontFamily:SERIF,fontSize:48,color:P.paper}}>{s[1]}</div><div style={{position:'absolute',left:0,right:0,top:68,height:2,background:`linear-gradient(90deg,${col},transparent)`,scale:`${p} 1`,transformOrigin:'left'}}/><div style={{position:'absolute',left:`${25+i*17}%`,top:52,width:34,height:34,border:`2px solid ${col}`,rotate:'45deg',background:P.night}}/></div>})}</div>
 <div style={{position:'absolute',left:96,right:96,bottom:112,fontFamily:SANS,fontSize:17,lineHeight:1.5,letterSpacing:2,color:P.mist,opacity:.75}}>{c.footer}</div>
 </Base>};

const Evidence:React.FC<FilmProps & {d:number}>=({locale,cover,d})=>{const f=useCurrentFrame(),c=C[locale],p=interpolate(f,[12,92],[0,1],{...clamp,easing:ease});return <Base cover={cover}>
 <div style={{position:'absolute',left:92,top:108}}><Label>{c.calc}</Label></div><div style={{position:'absolute',left:92,right:92,top:230,fontFamily:SERIF,fontSize:62,lineHeight:1.04,opacity:fade(f,d)}}>{c.evidence}<br/><span style={{fontStyle:'italic',color:P.gold}}>{c.evidence2}</span></div>
 <div style={{position:'absolute',left:72,right:72,top:760,height:720,border:`1px solid ${P.brass}66`,background:'rgba(2,8,13,.72)',boxShadow:'0 45px 100px rgba(0,0,0,.55)',opacity:p}}>
  <svg width="936" height="720"><line x1="88" y1="352" x2={88+760*p} y2="352" stroke={P.paper} strokeWidth="2"/><path d="M130 352c100-160 230-160 340 0s240 160 370 0" fill="none" stroke={P.gold} strokeWidth="4" opacity=".85"/><g fill={P.paper} fontFamily={SANS} fontSize="20"><text x="100" y="305">2102</text><text x="440" y="305">2002</text><text x="790" y="305">4099</text></g><g fill={P.mist} fontFamily={SERIF} fontSize="28"><text x="110" y="455">{locale==='fr'?'descente':'descent'}</text><text x="415" y="455">{locale==='fr'?'remontée':'ascent'}</text><text x="705" y="455">{locale==='fr'?'transition':'transition'}</text></g><circle cx="475" cy="352" r={18+10*Math.sin(f/10)} fill={P.gold}/></svg>
 </div></Base>};

const Wheel:React.FC<FilmProps & {d:number}>=({locale,cover,d})=>{const f=useCurrentFrame(),c=C[locale],p=interpolate(f,[0,90],[0,1],{...clamp,easing:ease}),rot=interpolate(f,[0,d],[8,-5]);const labels=locale==='fr'?['SATYA','TRETĀ','DVĀPARA','KALI']:['SATYA','TRETA','DVAPARA','KALI'];return <Base cover={cover}>
 <div style={{position:'absolute',left:92,top:110}}><Label>{c.wheel}</Label></div><div style={{position:'absolute',left:92,right:92,top:235,fontFamily:SERIF,fontSize:64,lineHeight:1.04,opacity:fade(f,d)}}>{c.wheelLead}</div>
 <svg width="1080" height="1920" style={{position:'absolute',inset:0,rotate:`${rot}deg`,opacity:p}}><g transform="translate(540 1120)"><circle r="365" fill="rgba(3,10,16,.7)" stroke={P.brass} strokeWidth="3"/><circle r="250" fill="none" stroke={P.ox} strokeWidth="2" strokeDasharray="8 16"/><path d="M-365 0H365M0-365V365" stroke={P.brass} strokeWidth="2" opacity=".55"/><path d="M0-365A365 365 0 0 1 365 0" fill="none" stroke={P.gold} strokeWidth="10" opacity=".7"/><path d="M365 0A365 365 0 0 1 0 365" fill="none" stroke={P.gold} strokeWidth="7" opacity=".5"/><path d="M0 365A365 365 0 0 1-365 0" fill="none" stroke={P.ox} strokeWidth="7" opacity=".55"/><path d="M-365 0A365 365 0 0 1 0-365" fill="none" stroke={P.ox} strokeWidth="10" opacity=".72"/>{labels.map((x,i)=><text key={x} x={[145,145,-250,-250][i]} y={[-150,190,190,-150][i]} fill={i<2?P.gold:P.mist} fontFamily={SANS} fontSize="28" letterSpacing="4">{x}</text>)}<text textAnchor="middle" y="-12" fill={P.paper} fontFamily={SERIF} fontSize="46">25 920</text><text textAnchor="middle" y="36" fill={P.mist} fontFamily={SANS} fontSize="17" letterSpacing="3">{locale==='fr'?'ANS':'YEARS'}</text></g></svg>
 </Base>};

const Method:React.FC<FilmProps & {d:number}>=({locale,cover,d})=>{const f=useCurrentFrame(),c=C[locale];return <Base cover={cover}>
 <div style={{position:'absolute',left:96,top:116}}><Label>{c.method}</Label></div><div style={{position:'absolute',left:96,right:96,top:275,fontFamily:SERIF,fontSize:78,lineHeight:1.02,opacity:fade(f,d)}}>{c.map}<br/><span style={{fontStyle:'italic',color:P.gold}}>{c.fog}</span></div>
 <div style={{position:'absolute',left:96,right:96,top:820,display:'grid',gridTemplateColumns:'1fr 1fr',gap:'70px 50px'}}>{c.cats.map((x,i)=>{const p=interpolate(f,[16+i*12,50+i*12],[0,1],clamp),col=[P.paper,P.ox,P.gold,P.signal][i];return <div key={x[0]} style={{opacity:p,borderTop:`2px solid ${col}`,paddingTop:24,minHeight:205,translate:`0 ${interpolate(p,[0,1],[32,0])}px`}}><div style={{fontFamily:SANS,fontSize:20,letterSpacing:3,color:col}}>{x[0]}</div><div style={{fontFamily:SERIF,fontSize:34,lineHeight:1.22,marginTop:18,color:P.mist}}>{x[1]}</div></div>})}</div>
 </Base>};

const Marsh:React.FC<FilmProps & {d:number}>=({locale,cover,d})=>{const f=useCurrentFrame(),c=C[locale],p=interpolate(f,[0,70],[0,1],{...clamp,easing:ease});return <Base cover={cover}>
 <svg width="1080" height="1920" style={{position:'absolute',inset:0}}><path d="M60 1340c150-270 360 70 550-220 145-225 295 55 430-190v820H60Z" fill={P.signal} opacity={.12*p}/><path d="M60 1340c150-270 360 70 550-220 145-225 295 55 430-190" fill="none" stroke={P.signal} strokeWidth="4" strokeDasharray="10 18" opacity={.75*p}/>{Array.from({length:17}).map((_,i)=><line key={i} x1={90+i*60} y1={1430-(i%4)*48} x2={155+i*60} y2={1505-(i%4)*48} stroke={P.signal} opacity={.36*p}/>)}</svg>
 <div style={{position:'absolute',left:92,right:92,top:300,whiteSpace:'pre-line',fontFamily:SERIF,fontSize:88,lineHeight:1.02,opacity:fade(f,d)}}>{c.no}</div><div style={{position:'absolute',left:92,right:92,bottom:190,fontFamily:SERIF,fontStyle:'italic',fontSize:42,lineHeight:1.25,color:P.mist}}>{c.marsh}</div>
 </Base>};

const Book:React.FC<{cover:string;progress:number;small?:boolean}>=({cover,progress,small=false})=>{const w=small?430:660,h=w*1.5,rot=small?0:interpolate(progress,[0,1],[-15,-5]),y=interpolate(progress,[0,1],[130,0]);return <div style={{position:'absolute',left:(1080-w)/2,top:small?280:305,width:w,height:h,perspective:1800,opacity:progress,translate:`0 ${y}px`}}><div style={{position:'absolute',inset:0,rotate:`${rot}deg`,filter:'drop-shadow(0 60px 80px rgba(0,0,0,.64))'}}><div style={{position:'absolute',inset:0,overflow:'hidden',border:`1px solid ${P.gold}77`,background:P.night}}><Img src={staticFile(cover)} style={{width:'100%',height:'100%',objectFit:'cover'}}/><div style={{position:'absolute',inset:0,background:'linear-gradient(115deg,transparent 30%,rgba(255,224,150,.15) 49%,transparent 68%)',translate:`${interpolate(progress,[0,1],[-700,650])}px 0`}}/></div>{!small&&<div style={{position:'absolute',left:-26,top:12,width:28,height:h-24,background:`linear-gradient(90deg,#010306,${P.slate})`,borderLeft:`1px solid ${P.brass}33`}}/>}</div></div>};
const Reveal:React.FC<FilmProps & {d:number}>=({locale,cover,d})=>{const f=useCurrentFrame(),c=C[locale],p=interpolate(f,[0,50],[0,1],{...clamp,easing:ease});return <Base cover={cover} grid={.07}><Book cover={cover} progress={p}/><div style={{position:'absolute',left:0,right:0,bottom:215,textAlign:'center',fontFamily:SANS,fontSize:21,letterSpacing:4,color:P.gold,opacity:fade(f,d)}}>{c.lucid}</div></Base>};
const End:React.FC<FilmProps & {d:number}>=({locale,cover,d})=>{const f=useCurrentFrame(),c=C[locale],p=interpolate(f,[0,28],[0,1],{...clamp,easing:ease});return <Base cover={cover} grid={.06}><Book cover={cover} progress={p} small/><div style={{position:'absolute',left:120,right:120,bottom:260,border:`1px solid ${P.brass}`,padding:'30px 35px',textAlign:'center',fontFamily:SANS,fontSize:23,letterSpacing:3.2,color:P.paper,opacity:p}}>{c.cta}</div><div style={{position:'absolute',left:0,right:0,bottom:170,textAlign:'center',fontFamily:SANS,fontSize:17,letterSpacing:4,color:P.gold,opacity:p}}>{c.author}</div></Base>};

const Film:React.FC<FilmProps>=({locale,cover})=><AbsoluteFill>
 <Sequence from={0} durationInFrames={120}><Coast locale={locale} cover={cover} d={120}/></Sequence>
 <Sequence from={120} durationInFrames={150}><Chronologies locale={locale} cover={cover} d={150}/></Sequence>
 <Sequence from={270} durationInFrames={165}><Evidence locale={locale} cover={cover} d={165}/></Sequence>
 <Sequence from={435} durationInFrames={150}><Wheel locale={locale} cover={cover} d={150}/></Sequence>
 <Sequence from={585} durationInFrames={165}><Method locale={locale} cover={cover} d={165}/></Sequence>
 <Sequence from={750} durationInFrames={105}><Marsh locale={locale} cover={cover} d={105}/></Sequence>
 <Sequence from={855} durationInFrames={165}><Reveal locale={locale} cover={cover} d={165}/></Sequence>
 <Sequence from={1020} durationInFrames={60}><End locale={locale} cover={cover} d={60}/></Sequence>
 <Audio src={staticFile('audio/yuga-final.wav')} volume={.9}/>
</AbsoluteFill>;

const Poster:React.FC<FilmProps>=({locale,cover})=>{const c=C[locale];return <Base cover={cover} grid={.06}><Book cover={cover} progress={1} small/><div style={{position:'absolute',left:120,right:120,bottom:260,border:`1px solid ${P.brass}`,padding:'30px 35px',textAlign:'center',fontFamily:SANS,fontSize:23,letterSpacing:3.2,color:P.paper}}>{c.cta}</div><div style={{position:'absolute',left:0,right:0,bottom:170,textAlign:'center',fontFamily:SANS,fontSize:17,letterSpacing:4,color:P.gold}}>{c.author}</div></Base>};
const Root=()=> <><Composition id="YugaFinalFR" component={Film} durationInFrames={1080} fps={30} width={1080} height={1920} defaultProps={{locale:'fr' as const,cover:'assets/cover-fr.jpg'}}/><Composition id="YugaFinalEN" component={Film} durationInFrames={1080} fps={30} width={1080} height={1920} defaultProps={{locale:'en' as const,cover:'assets/cover-en.png'}}/><Still id="YugaPosterFR" component={Poster} width={1080} height={1920} defaultProps={{locale:'fr' as const,cover:'assets/cover-fr.jpg'}}/><Still id="YugaPosterEN" component={Poster} width={1080} height={1920} defaultProps={{locale:'en' as const,cover:'assets/cover-en.png'}}/></>;
registerRoot(Root);
