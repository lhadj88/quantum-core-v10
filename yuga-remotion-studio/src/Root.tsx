import React from 'react';
import {Composition, Still} from 'remotion';
import {StudioFilm} from './StudioFilm';

export const RemotionRoot: React.FC = () => <>
  <Composition id="YugaStudioFR" component={StudioFilm} durationInFrames={1080} fps={30} width={1080} height={1920} defaultProps={{lang:'fr' as const}}/>
  <Composition id="YugaStudioEN" component={StudioFilm} durationInFrames={1080} fps={30} width={1080} height={1920} defaultProps={{lang:'en' as const}}/>
  <Still id="YugaPosterFR" component={StudioFilm} width={1080} height={1920} defaultProps={{lang:'fr' as const}}/>
  <Still id="YugaPosterEN" component={StudioFilm} width={1080} height={1920} defaultProps={{lang:'en' as const}}/>
</>;
