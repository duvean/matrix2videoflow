import { useMemo, useRef, useState } from 'react';
import GridLayout, { type Layout } from 'react-grid-layout';
import { Eye, EyeOff, Layers2, MousePointer2, Play, Plus, Scissors, SkipBack, SkipForward } from 'lucide-react';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';

type Mask = {
  id: string;
  name: string;
  selected: boolean;
  visible: boolean;
};

const INITIAL_LAYOUT: Layout[] = [
  { i: 'batches', x: 0, y: 0, w: 3, h: 12 },
  { i: 'preview', x: 3, y: 0, w: 7, h: 12 },
  { i: 'nodegraph', x: 10, y: 0, w: 4, h: 7 },
  { i: 'masks', x: 10, y: 7, w: 4, h: 5 },
  { i: 'timeline', x: 0, y: 12, w: 14, h: 4 }
];

const mockBatches = Array.from({ length: 6 }).map((_, i) => `batch_${String(i + 1).padStart(2, '0')}.jfif`);
const mockFrames = Array.from({ length: 30 }).map((_, i) => i + 1);

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const timelineRef = useRef<HTMLDivElement>(null);

  const [layout, setLayout] = useState<Layout[]>(INITIAL_LAYOUT);
  const [progress, setProgress] = useState(0);
  const [masks, setMasks] = useState<Mask[]>([
    { id: 'm1', name: 'mask 1', selected: true, visible: true },
    { id: 'm2', name: 'mask 2', selected: false, visible: true },
    { id: 'm3', name: 'mask 3', selected: false, visible: false }
  ]);

  const selectedMasks = useMemo(() => masks.filter((m) => m.selected), [masks]);

  const onTimeUpdate = () => {
    const video = videoRef.current;
    const timeline = timelineRef.current;
    if (!video || !timeline || video.duration === 0) return;

    const ratio = video.currentTime / video.duration;
    const maxScroll = timeline.scrollWidth - timeline.clientWidth;
    timeline.scrollLeft = maxScroll * ratio;
  };

  const handleSpaceRestart = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.code !== 'Space') return;
    event.preventDefault();
    const video = videoRef.current;
    if (!video) return;

    if (video.ended || video.currentTime >= video.duration - 0.05) {
      video.currentTime = 0;
    }

    void video.play();
  };

  const toggleMask = (id: string, key: 'selected' | 'visible') => {
    setMasks((prev) => prev.map((m) => (m.id === id ? { ...m, [key]: !m[key] } : m)));
  };

  const addMask = () => {
    const nextId = `m${masks.length + 1}`;
    setMasks((prev) => [...prev, { id: nextId, name: `mask ${prev.length + 1}`, selected: true, visible: true }]);
  };

  const runInpaint = () => {
    const names = selectedMasks.map((m) => m.name).join(', ');
    window.alert(`Inpaint for selected masks only: ${names || 'none'}`);
    setProgress(35);
    setTimeout(() => setProgress(100), 450);
    setTimeout(() => setProgress(0), 1400);
  };

  return (
    <div className="app" onKeyDown={handleSpaceRestart} tabIndex={0}>
      <header className="topbar">
        <h1>Matrix2VideoFlow • Tauri UI</h1>
      </header>

      <GridLayout
        className="layout"
        cols={14}
        rowHeight={36}
        width={1360}
        margin={[10, 10]}
        draggableHandle=".panel-title"
        layout={layout}
        onLayoutChange={setLayout}
      >
        <section key="batches" className="panel">
          <div className="panel-title"><Layers2 size={14} /> Batches</div>
          <button className="btn btn-ghost">+ Add Batch</button>
          <ul className="batch-list">
            {mockBatches.map((batch) => (
              <li key={batch}>{batch}</li>
            ))}
          </ul>
        </section>

        <section key="preview" className="panel">
          <div className="panel-title"><Play size={14} /> Video Preview</div>
          <video
            ref={videoRef}
            className="video"
            controls
            onTimeUpdate={onTimeUpdate}
            src="https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4"
          />
          <div className="controls">
            <button className="icon-btn"><SkipBack size={14} /></button>
            <button className="icon-btn"><Play size={14} /></button>
            <button className="icon-btn"><SkipForward size={14} /></button>
            <button className="btn" onClick={() => setProgress(65)}>Export Video</button>
          </div>
          <progress max={100} value={progress} className="global-progress" />
        </section>

        <section key="nodegraph" className="panel">
          <div className="panel-title"><Scissors size={14} /> Processing Graph</div>
          <div className="nodegraph">
            <div className="node">Input Frames</div>
            <span>→</span>
            <div className="node">Mask Compose (selected only)</div>
            <span>→</span>
            <div className="node">Inpaint / FILM / CV</div>
            <span>→</span>
            <div className="node">Output</div>
          </div>
          <p className="muted">Показывает активные шаги и маски, которые реально участвуют в пайплайне.</p>
        </section>

        <section key="masks" className="panel">
          <div className="panel-title"><MousePointer2 size={14} /> Masks</div>
          <ul className="mask-list">
            {masks.map((mask) => (
              <li key={mask.id}>
                <button className={`icon-btn ${mask.selected ? 'active' : ''}`} onClick={() => toggleMask(mask.id, 'selected')}>
                  <MousePointer2 size={14} />
                </button>
                <span>{mask.name}</span>
                <button className="icon-btn" onClick={() => toggleMask(mask.id, 'visible')}>
                  {mask.visible ? <Eye size={14} /> : <EyeOff size={14} />}
                </button>
              </li>
            ))}
          </ul>
          <div className="actions-row">
            <button className="btn btn-ghost" onClick={addMask}><Plus size={14} /> New Mask</button>
            <button className="btn" onClick={runInpaint}>Inpaint selected frames</button>
          </div>
        </section>

        <section key="timeline" className="panel">
          <div className="panel-title">Timeline</div>
          <div className="timeline" ref={timelineRef}>
            {mockFrames.map((frame) => (
              <div className="frame" key={frame}>
                <div className="thumb" />
                <span>{String(frame).padStart(2, '0')}</span>
              </div>
            ))}
          </div>
        </section>
      </GridLayout>
    </div>
  );
}
