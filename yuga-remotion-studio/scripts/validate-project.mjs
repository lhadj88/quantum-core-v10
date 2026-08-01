import fs from 'node:fs';
import path from 'node:path';
const must=[
  'package.json','src/index.ts','src/Root.tsx','src/StudioFilm.tsx','src/content.ts',
  'src/components/FilmPrimitives.tsx','public/assets/starfield.svg','public/assets/grain.svg','public/assets/paper.svg','public/audio/yuga-studio-master.wav'
];
for(const f of must){if(!fs.existsSync(path.resolve(f))) throw new Error(`Missing ${f}`)}
const src=fs.readFileSync('src/StudioFilm.tsx','utf8');
for(const forbidden of ['@keyframes','transition:','animation:']){if(src.includes(forbidden)) throw new Error(`Forbidden CSS animation token: ${forbidden}`)}
const wav=fs.statSync('public/audio/yuga-studio-master.wav');
if(wav.size<1000000) throw new Error('Audio master unexpectedly small');
console.log('Project validation passed.');
