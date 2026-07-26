import React from 'react';
import {AbsoluteFill, Easing, Img, interpolate, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';

export const palette = {
  night: '#020713',
  deep: '#07101d',
  brass: '#e9ad42',
  brassSoft: '#b7802f',
  ivory: '#f0e5cf',
  mist: '#95a7b8',
  blue: '#355f88',
  rust: '#7f4935',
};

export const Scene: React.FC<React.PropsWithChildren<{duration: number; lead?: number; trail?: number; zoom?: number}>> = ({children,duration,lead=18,trail=18,zoom=0.025}) => {
  const frame=useCurrentFrame();
  const opacity=interpolate(frame,[0,lead,duration-trail,duration],[0,1,1,0],{extrapolateLeft:'clamp',extrapolateRight:'clamp',easing:Easing.bezier(.22,1,.36,1)});
  const scale=interpolate(frame,[0,duration],[1+zoom,1],{extrapolateLeft:'clamp',extrapolateRight:'clamp'});
  return <AbsoluteFill style={{opacity,scale,backgroundColor:palette.night,overflow:'hidden'}}>{children}</AbsoluteFill>;
};

export const DeepSpace: React.FC<{intensity?: number; horizon?: boolean}> = ({intensity=1,horizon=false}) => {
  const frame=useCurrentFrame();
  const {durationInFrames}=useVideoConfig();
  return <AbsoluteFill>
    <Img src={staticFile('assets/starfield.svg')} style={{width:'100%',height:'100%',objectFit:'cover',scale:interpolate(frame,[0,durationInFrames],[1.07,1.0]),translate:`${interpolate(frame,[0,durationInFrames],[-14,14])}px ${interpolate(frame,[0,durationInFrames],[-20,10])}px`,opacity:.94}}/>
    <AbsoluteFill style={{background:`radial-gradient(circle at 75% 20%, rgba(53,95,136,${.22*intensity}), transparent 38%), radial-gradient(circle at 46% 74%, rgba(233,173,66,${.16*intensity}), transparent 30%), linear-gradient(180deg, rgba(0,0,0,.02), rgba(1,5,13,.68))`}}/>
    {horizon ? <AbsoluteFill style={{top:'63%',height:'37%',background:'radial-gradient(ellipse at 50% 0%, rgba(243,174,63,.22), transparent 28%), linear-gradient(180deg, rgba(4,12,25,.1), rgba(0,3,10,.92))',borderTop:'1px solid rgba(233,173,66,.22)'}}/>:null}
  </AbsoluteFill>;
};

export const Grain: React.FC = () => {
  const frame=useCurrentFrame();
  return <AbsoluteFill style={{pointerEvents:'none',opacity:.11,mixBlendMode:'soft-light',backgroundImage:`url(${staticFile('assets/grain.svg')})`,backgroundRepeat:'repeat',backgroundSize:'380px 380px',translate:`${(frame*17)%380-190}px ${(frame*11)%380-190}px`}}/>;
};

export const TitleBlock: React.FC<{kicker?:string; title:string; subtitle?:string; align?:'left'|'center'; maxWidth?:number}> = ({kicker,title,subtitle,align='left',maxWidth=900}) => {
  const frame=useCurrentFrame();
  const enter=interpolate(frame,[0,28],[0,1],{extrapolateRight:'clamp',easing:Easing.bezier(.16,1,.3,1)});
  return <div style={{maxWidth,textAlign:align,opacity:enter,translate:`0 ${interpolate(enter,[0,1],[55,0])}px`}}>
    {kicker ? <div style={{fontFamily:'Arial, sans-serif',fontSize:22,letterSpacing:8,color:palette.brass,marginBottom:20,fontWeight:600}}>{kicker}</div>:null}
    <div style={{fontFamily:'Georgia, Times New Roman, serif',fontSize:74,lineHeight:.98,letterSpacing:-1.2,color:palette.ivory,textShadow:'0 8px 40px rgba(0,0,0,.55)'}}>{title}</div>
    {subtitle ? <div style={{fontFamily:'Arial, sans-serif',fontSize:27,lineHeight:1.45,letterSpacing:.7,color:'rgba(240,229,207,.72)',marginTop:24}}>{subtitle}</div>:null}
  </div>;
};

export const BrassRule: React.FC<{width?:number}> = ({width=520}) => {
  const frame=useCurrentFrame();
  const p=interpolate(frame,[0,36],[0,1],{extrapolateRight:'clamp',easing:Easing.bezier(.16,1,.3,1)});
  return <div style={{height:1,width:width*p,background:'linear-gradient(90deg, transparent, #e9ad42 16%, #ffe3a0 50%, #e9ad42 84%, transparent)',boxShadow:'0 0 20px rgba(233,173,66,.55)'}}/>;
};

export const Book3D: React.FC<{cover:'fr'|'en'}> = ({cover}) => {
  const frame=useCurrentFrame();
  const turn=interpolate(frame,[0,75,160],[32,3,-7],{extrapolateLeft:'clamp',extrapolateRight:'clamp',easing:Easing.bezier(.16,1,.3,1)});
  const lift=interpolate(frame,[0,90],[110,0],{extrapolateRight:'clamp',easing:Easing.bezier(.16,1,.3,1)});
  const glow=interpolate(frame,[20,100,180],[.05,.6,.2],{extrapolateLeft:'clamp',extrapolateRight:'clamp'});
  const fr=cover==='fr';
  const title=fr?'LA\nGRANDE\nTRANSITION':'THE\nGREAT\nTRANSITION';
  const subtitle=fr?'Une carte lucide du cycle des Yugas':'A Lucid Map of the Yuga Cycle';
  return <div style={{width:470,height:705,perspective:1800,translate:`0 ${lift}px`}}>
    <div style={{position:'relative',width:'100%',height:'100%',transformStyle:'preserve-3d',transform:`rotate(-2deg) rotateY(${turn}deg)`,boxShadow:`0 45px 120px rgba(0,0,0,.72), 0 0 95px rgba(233,173,66,${glow})`}}>
      <div style={{position:'absolute',inset:0,backfaceVisibility:'hidden',overflow:'hidden',borderRadius:3,border:'1px solid rgba(255,224,150,.35)',background:'#07101d'}}>
        <Img src={staticFile('assets/starfield.svg')} style={{position:'absolute',width:'100%',height:'100%',objectFit:'cover',scale:1.05}}/>
        <div style={{position:'absolute',inset:0,background:'linear-gradient(180deg,rgba(0,3,10,.06),rgba(0,3,10,.25) 60%,rgba(0,3,10,.8))'}}/>
        <svg width="470" height="705" viewBox="0 0 470 705" style={{position:'absolute',inset:0}}>
          <defs><filter id="cg"><feGaussianBlur stdDeviation="4" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
          <path d="M -60 390 Q 235 610 530 390" fill="none" stroke="#f6b94c" strokeWidth="2.8" filter="url(#cg)"/>
          <path d="M -20 555 Q 235 500 490 555" fill="none" stroke="#ffd66f" strokeWidth="3.2" filter="url(#cg)"/>
          <circle cx="235" cy="547" r="14" fill="#ffd66f" opacity=".9" filter="url(#cg)"/>
        </svg>
        <div style={{position:'absolute',left:32,right:32,top:50,textAlign:'center',whiteSpace:'pre-line',fontFamily:'Georgia,serif',fontSize:fr?43:45,lineHeight:.92,letterSpacing:2,color:'#efb445',textShadow:'0 2px 18px rgba(0,0,0,.75)'}}>{title}</div>
        <div style={{position:'absolute',left:35,right:35,top:250,textAlign:'center',fontFamily:'Georgia,serif',fontSize:30,letterSpacing:2,color:'#f5c762'}}>2044–2082</div>
        <div style={{position:'absolute',left:35,right:35,top:294,textAlign:'center',fontFamily:'Georgia,serif',fontSize:15,lineHeight:1.25,color:'#f0c36b'}}>{subtitle}</div>
        <div style={{position:'absolute',left:20,right:20,bottom:28,textAlign:'center',fontFamily:'Georgia,serif',fontSize:17,letterSpacing:1.6,color:'#efb445'}}>THE LUCID CARTOGRAPHER</div>
        <div style={{position:'absolute',inset:0,background:'linear-gradient(105deg, rgba(255,255,255,.16), transparent 19%, transparent 70%, rgba(255,200,80,.08))',mixBlendMode:'screen'}}/>
      </div>
      <div style={{position:'absolute',left:-32,top:4,width:32,height:'calc(100% - 8px)',background:'linear-gradient(90deg,#211509,#60411c,#171007)',transformOrigin:'right center',transform:'rotateY(-90deg)',boxShadow:'inset -2px 0 0 rgba(255,224,150,.2)'}}/>
      <div style={{position:'absolute',right:-26,top:7,width:26,height:'calc(100% - 14px)',background:'repeating-linear-gradient(90deg,#e4d9c4 0 2px,#c2b69f 2px 3px)',transformOrigin:'left center',transform:'rotateY(90deg)'}}/>
      <div style={{position:'absolute',left:7,bottom:-26,width:'calc(100% - 14px)',height:26,background:'repeating-linear-gradient(0deg,#e4d9c4 0 2px,#c2b69f 2px 3px)',transformOrigin:'center top',transform:'rotateX(90deg)'}}/>
    </div>
  </div>;
};
