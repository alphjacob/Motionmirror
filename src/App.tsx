import { type CSSProperties, useMemo, useRef, useState } from 'react';
import {
  Activity,
  Aperture,
  Camera,
  Check,
  ChevronDown,
  CircleHelp,
  Crosshair,
  Download,
  Focus,
  LockKeyhole,
  Maximize2,
  Pause,
  Play,
  RotateCcw,
  Save,
  SlidersHorizontal,
  Sparkles,
  Square,
  Trash2,
  Video,
} from 'lucide-react';
import {
  DemoAvatar as RiggedDemoAvatar,
  type AvatarOutfit,
} from '@/avatar/DemoAvatar';
import { legacyImagePointToWorld } from '@/avatar/retarget';
import { estimateEmotion, usePoseTracking } from '@/tracking/usePoseTracking';
import { useMotionRecorder } from '@/recording/useMotionRecorder';
import { mapLandmarkToCoverPreview } from '@/avatar/retarget';
import type { MotionRecording } from '@/recording/types';
import type {
  FaceCues,
  PoseCalibration,
  PoseLandmark,
  TrackedHand,
} from '@/tracking/types';

type Outfit = { id: AvatarOutfit; name: string; color: string; shade: string };

const outfits: Outfit[] = [
  { id: 'mocap', name: 'Motion-capture', color: '#68c7a8', shade: '#377b68' },
  { id: 'casual', name: 'Casual', color: '#668ce8', shade: '#3d5fae' },
  { id: 'future', name: 'Futuristic', color: '#5ce0cc', shade: '#285d6b' },
];

const POSE_CONNECTIONS: Array<[number, number]> = [
  [0, 11], [0, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28],
];

function DemoAvatar({
  outfit,
  landmarks,
  calibration,
  hands,
  faceCues,
  active,
  transparent = false,
  className = '',
  style,
}: {
  outfit: Outfit;
  landmarks: PoseLandmark[];
  calibration: PoseCalibration | null;
  hands: TrackedHand[];
  faceCues: FaceCues | null;
  active: boolean;
  transparent?: boolean;
  className?: string;
  style?: CSSProperties;
}) {
  return (
    <div
      className={`demo-avatar-scene ${className}`}
      style={style}
      role="img"
      aria-label={`Configurable demo avatar wearing ${outfit.name}`}
    >
      <RiggedDemoAvatar
        landmarks={landmarks}
        calibration={calibration}
        hands={hands}
        faceCues={faceCues}
        outfit={outfit.id}
        active={active}
        transparent={transparent}
      />
    </div>
  );
}

function statusLabel(status: string) {
  if (status === 'tracking') return 'Tracking movement';
  if (status === 'loading') return 'Opening camera';
  if (status === 'lost') return 'Pose not in frame';
  if (status === 'error') return 'Camera unavailable';
  return 'Camera paused';
}

function formatDuration(milliseconds: number) {
  const seconds = Math.max(0, Math.floor(milliseconds / 1000));
  return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
}

function App() {
  const [sensitivity, setSensitivity] = useState(68);
  const {
    videoRef,
    cameraReady,
    status,
    confidence,
    landmarks,
    avatarLandmarks,
    calibration,
    hands,
    faceCues,
    error,
    featureWarning,
    startTracking,
    stopTracking,
    calibrate,
  } = usePoseTracking(sensitivity);
  const [avatarName, setAvatarName] = useState<string>(() => {
    try {
      return sessionStorage.getItem('motionmirror_avatar_name') || '';
    } catch {
      return '';
    }
  });
  const [isNamingOpen, setIsNamingOpen] = useState(false);
  const [namingInput, setNamingInput] = useState('');

  const handleStartCameraClick = () => {
    if (avatarName.trim()) {
      void startTracking();
    } else {
      setNamingInput('');
      setIsNamingOpen(true);
    }
  };

  const handleConfirmName = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const trimmed = namingInput.trim();
    if (!trimmed) return;
    setAvatarName(trimmed);
    try {
      sessionStorage.setItem('motionmirror_avatar_name', trimmed);
    } catch {
      // Ignore storage error
    }
    setIsNamingOpen(false);
    void startTracking();
  };

  const handleCancelNaming = () => {
    setIsNamingOpen(false);
  };

  const [outfit, setOutfit] = useState(outfits[0]);
  const [showSkeleton, setShowSkeleton] = useState(true);
  const [showDemoAvatar, setShowDemoAvatar] = useState(true);
  const cameraSurfaceRef = useRef<HTMLDivElement>(null);
  const recorder = useMotionRecorder({
    landmarks,
    avatarLandmarks,
    hands,
    faceCues,
    confidence,
    tracking: status === 'tracking',
    outfit: outfit.id,
  });

  const points = useMemo(() => {
    return landmarks.filter(
      (point): point is PoseLandmark =>
        Boolean(point) &&
        typeof point.x === 'number' &&
        typeof point.y === 'number',
    );
  }, [landmarks]);
  const surfaceRect = cameraSurfaceRef.current?.getBoundingClientRect();
  const cameraWidth = videoRef.current?.videoWidth ?? 0;
  const cameraHeight = videoRef.current?.videoHeight ?? 0;
  const toCameraPreviewPoint = (point: PoseLandmark) => {
    const mapped = mapLandmarkToCoverPreview(
      point,
      cameraWidth,
      cameraHeight,
      surfaceRect?.width ?? 0,
      surfaceRect?.height ?? 0,
    );
    return { ...point, ...mapped };
  };
  const previewPoints = points.map(toCameraPreviewPoint);
  const previewHands = hands.map((hand) => ({
    ...hand,
    landmarks: hand.landmarks.map(toCameraPreviewPoint),
  }));
  const displayedFrame = recorder.playbackFrame;
  const displayedHands = displayedFrame?.hands ?? hands;
  const displayedFaceCues = displayedFrame?.faceCues ?? faceCues;
  const displayedOutfit =
    outfits.find((item) => item.id === recorder.playbackRecording?.outfit) ??
    outfit;
  const isRecording =
    recorder.phase === 'recording' || recorder.phase === 'recording-paused';
  const isPlayback =
    recorder.phase === 'playing' || recorder.phase === 'playback-paused';
  const displayedAvatarPoints = displayedFrame
    ? displayedFrame.worldLandmarks ??
      displayedFrame.landmarks.map(legacyImagePointToWorld)
    : avatarLandmarks;
  const displayedCalibration = isPlayback ? null : calibration;
  const emotionEstimate = displayedFaceCues?.emotion ?? (displayedFaceCues ? estimateEmotion(displayedFaceCues) : null);
  const confidenceValue = Number.isFinite(confidence) ? Math.max(0, Math.min(1, Number(confidence))) : 0;
  const confidencePercent = Math.round(confidenceValue * 100);
  const recordTimeLabel = formatDuration(recorder.elapsedMs);
  const playbackTimeLabel = formatDuration(recorder.playbackTimeMs);
  const playbackDurationLabel = formatDuration(recorder.playbackDurationMs);

  const deleteRecording = async (recording: MotionRecording) => {
    const confirmed = window.confirm(
      `Delete “${recording.name}”? This removes the saved motion data from this browser.`,
    );
    if (confirmed) await recorder.deleteRecording(recording.id);
  };

  const exportRecording = (recording: MotionRecording) => {
    const blob = new Blob([JSON.stringify(recording, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${recording.id}.motionmirror.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="studio-shell">
      {isNamingOpen && (
        <div className="avatar-modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="modal-naming-heading" data-testid="modal-avatar-naming">
          <div className="avatar-modal-card">
            <div className="avatar-modal-header">
              <div className="avatar-modal-icon"><Sparkles size={18} /></div>
              <h2 id="modal-naming-heading">Name your Avatar</h2>
              <p>Give your 3D motion avatar a personal name for this session before activating the camera.</p>
            </div>
            <form onSubmit={handleConfirmName}>
              <div className="avatar-modal-input-wrap">
                <label htmlFor="avatar-name-input" className="sr-only">Avatar Name</label>
                <input
                  id="avatar-name-input"
                  type="text"
                  className="avatar-modal-input"
                  placeholder="e.g. Ken, Maya, Nova..."
                  value={namingInput}
                  onChange={(e) => setNamingInput(e.target.value)}
                  maxLength={32}
                  autoFocus
                  data-testid="input-avatar-name"
                />
              </div>
              <div className="avatar-modal-actions">
                <button
                  type="button"
                  className="secondary-button"
                  onClick={handleCancelNaming}
                  data-testid="button-cancel-naming"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="primary-button"
                  disabled={!namingInput.trim()}
                  data-testid="button-confirm-naming"
                >
                  Confirm & Start camera
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true"><Aperture size={19} strokeWidth={2.1} /></div>
          <span className="brand-name">motionmirror</span>
          <span className="brand-caption">creator studio</span>
        </div>
        <div className="topbar-right">
          <div className="local-pill" data-testid="status-local-processing"><span /> Processing stays in this browser</div>
          <button className="icon-button" type="button" aria-label="Studio help" data-testid="button-help"><CircleHelp size={16} /></button>
        </div>
      </header>

      <main className="workspace">
        <div className="page-heading">
          <div>
            <div className="eyebrow">Workspace / motion capture</div>
            <h1>Motion studio</h1>
            <p>Find your frame. Tune the avatar. Make a take.</p>
          </div>
          <div className="workspace-meta"><span>PROJECT</span><strong>Untitled session</strong><ChevronDown size={13} /></div>
        </div>

        <div className="studio-grid">
          <section className="main-column" aria-label="Motion capture workspace">
            <div className="panel stage-panel">
              <div className="panel-heading">
                <div>
                  <div className="panel-title"><Focus size={15} /> Capture stage</div>
                  <div className="panel-subtitle" style={{ marginTop: 4 }}>LIVE VIEW · {statusLabel(status).toUpperCase()}</div>
                </div>
                <div className="stage-actions">
                  <button type="button" className={`small-control ${showSkeleton ? 'active' : ''}`} aria-pressed={showSkeleton} onClick={() => setShowSkeleton((value) => !value)} data-testid="toggle-skeleton"><Activity size={13} /> Skeleton</button>
                  <button type="button" className={`small-control ${showDemoAvatar ? 'active' : ''}`} aria-pressed={showDemoAvatar} onClick={() => setShowDemoAvatar((value) => !value)} data-testid="toggle-demo-avatar"><Sparkles size={13} /> Avatar</button>
                  <button type="button" className="small-control" aria-label="Recenter stage" onClick={() => { if (status === 'tracking') calibrate(); }} data-testid="button-recenter"><Maximize2 size={13} /></button>
                </div>
              </div>
              <div className="stage-area" data-testid="status-capture-stage">
                {showDemoAvatar ? (
                  <DemoAvatar
                    outfit={displayedOutfit}
                    landmarks={displayedAvatarPoints}
                    calibration={displayedCalibration}
                    hands={displayedHands}
                    faceCues={displayedFaceCues}
                    active={status === 'tracking' || isPlayback}
                    transparent
                    className="avatar-art"
                  />
                ) : (
                  <div className="avatar-hidden-note">Avatar preview hidden</div>
                )}
                {showDemoAvatar && emotionEstimate && (status === 'tracking' || isPlayback) && (
                  <div className="stage-emotion-badge" data-testid="badge-emotion-estimate">
                    <span className="emotion-icon" aria-hidden="true">{emotionEstimate.icon}</span>
                    <div className="emotion-meta">
                      <span className="emotion-status-text">
                        {`${avatarName.trim() || 'Avatar'} is ${emotionEstimate.emotion.toLowerCase()}`}
                      </span>
                      <span className="emotion-confidence">{Math.round(emotionEstimate.confidence * 100)}% cue confidence</span>
                    </div>
                  </div>
                )}
                <div className="stage-camera-preview">
                  <div className="camera-preview-heading">
                    <span>Camera preview</span>
                    <i className={cameraReady ? 'connected' : ''} />
                  </div>
                  <div className="camera-preview-surface" ref={cameraSurfaceRef}>
                    <video
                      ref={videoRef}
                      className="preview-video"
                      autoPlay
                      muted
                      playsInline
                      aria-label="Live camera preview"
                      style={{ display: cameraReady ? 'block' : 'none' }}
                    />
                    {(status === 'tracking' || status === 'lost') && showSkeleton && (
                      <svg className="tracking-overlay" viewBox="0 0 1 1" preserveAspectRatio="none" aria-label="Live pose and hand landmarks" data-testid="status-landmark-overlay">
                        <g fill="none" stroke="#d6f270" strokeWidth=".004" strokeLinecap="round" vectorEffect="non-scaling-stroke">
                          {POSE_CONNECTIONS.map(([from, to]) => {
                            const start = previewPoints[from];
                            const end = previewPoints[to];
                            if (!start || !end) return null;
                            if ((start.visibility ?? 1) < 0.2 || (end.visibility ?? 1) < 0.2) return null;
                            return <line key={`pose-${from}-${to}`} x1={start.x} y1={start.y} x2={end.x} y2={end.y} />;
                          })}
                          {previewHands.map((hand, handIndex) =>
                            [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]].map(([from, to]) => {
                              const start = hand.landmarks[from];
                              const end = hand.landmarks[to];
                              return start && end ? <line key={`hand-${handIndex}-${from}-${to}`} x1={start.x} y1={start.y} x2={end.x} y2={end.y} /> : null;
                            }),
                          )}
                        </g>
                        {previewPoints.map((point, index) => {
                          const visible = point.visibility === undefined || point.visibility > 0.25;
                          return visible ? <circle key={index} cx={point.x} cy={point.y} r=".007" fill="#d6f270" stroke="#293326" strokeWidth=".0022" vectorEffect="non-scaling-stroke" /> : null;
                        })}
                      </svg>
                    )}
                    {!cameraReady && (
                      <div className="camera-empty">
                        <div className="camera-symbol"><Camera size={18} /></div>
                        <strong>{status === 'loading' ? 'Preparing camera and trackers' : status === 'error' ? 'Camera needs attention' : 'Camera is off'}</strong>
                        <p>{status === 'error' ? error ?? 'Check camera permission and try again.' : 'Start the camera when ready. Video stays on this device.'}</p>
                      </div>
                    )}
                  </div>
                </div>
                <div className="stage-corner"><span className="live-dot" /> {isPlayback ? 'Saved motion' : status === 'tracking' ? 'Live avatar' : 'Avatar preview'}</div>
                <div className="stage-controls">
                  <div className="stage-legend"><b>POSE</b> &nbsp; Keep torso, shoulders and hands in view<br /><b>PRIVACY</b> &nbsp; Camera frames remain in this browser</div>
                  <div className={`capture-state ${status === 'tracking' || isPlayback ? 'tracking' : ''}`} data-testid="status-tracking"><i />{isPlayback ? 'Playing saved motion' : statusLabel(status)}</div>
                </div>
              </div>
              <div className="action-row">
                {status === 'tracking' || status === 'lost' ? (
                  <button className="primary-button" type="button" onClick={stopTracking} data-testid="button-stop-camera"><Pause size={14} /> Pause camera</button>
                ) : (
                  <button className="primary-button" type="button" onClick={handleStartCameraClick} disabled={status === 'loading'} data-testid="button-start-camera"><Play size={14} /> {status === 'loading' ? 'Connecting…' : 'Start camera'}</button>
                )}
                <button className="secondary-button" type="button" onClick={calibrate} disabled={status !== 'tracking'} data-testid="button-calibrate"><Crosshair size={14} /> Calibrate</button>
                <div className="action-help"><LockKeyhole size={12} /> Camera access is browser-local</div>
              </div>
              {error && <div className="error-message" role="alert" data-testid="status-camera-error">{String(error)}</div>}
              {featureWarning && <div className="feature-warning" role="status" data-testid="status-feature-warning">{featureWarning}</div>}
            </div>

            <div className="metrics-row">
              <div className="panel metric-card">
                <div className="metric-top"><span>Pose confidence</span><Activity size={13} /></div>
                <div className="metric-value" data-testid="text-confidence">{status === 'tracking' ? `${confidencePercent}` : '—'}<small>{status === 'tracking' ? '%' : 'waiting for camera'}</small></div>
                <div className="meter" aria-label={`Pose confidence ${confidencePercent}%`}><span style={{ width: `${status === 'tracking' ? confidencePercent : 0}%` }} /></div>
              </div>
              <div className="panel metric-card">
                <div className="metric-top"><span>Landmarks seen</span><Crosshair size={13} /></div>
                <div className="metric-value" data-testid="text-landmark-count">{status === 'tracking' ? points.length : '—'}<small>{status === 'tracking' ? 'points' : 'not active'}</small></div>
              </div>
              <div className="panel metric-card">
                <div className="metric-top"><span>Hands tracked</span><Video size={13} /></div>
                <div className="metric-value" data-testid="text-hands-tracked">{status === 'tracking' || isPlayback ? displayedHands.length : '—'}<small>{status === 'tracking' || isPlayback ? 'hands' : 'not active'}</small></div>
                <div className="panel-subtitle" style={{ marginTop: 4 }}>UP TO 2 HANDS</div>
              </div>
            </div>
          </section>

          <aside className="side-column" aria-label="Studio settings">
            <section className="panel side-panel">
              <div className="side-head"><h2>Avatar appearance</h2><span>DEMO</span></div>
              <div className="avatar-preview">
                <DemoAvatar outfit={displayedOutfit} landmarks={displayedAvatarPoints} calibration={displayedCalibration} hands={displayedHands} faceCues={displayedFaceCues} active={status === 'tracking' || isPlayback} transparent className="avatar-mini" />
                <span className="demo-label">CONFIGURABLE DEMO AVATAR</span>
              </div>
              <div className="avatar-name">{avatarName.trim() || 'Studio figure'} <span style={{ color: '#92968c', fontWeight: 500 }}>· {displayedOutfit.name}</span></div>
              <p className="avatar-note">A procedural demo figure, not a reconstructed likeness. Presets adjust its materials and tracking markers; rigged wardrobe meshes are not included.</p>
              <div className="swatches" role="group" aria-label="Choose demo avatar outfit">
                {outfits.map((item) => (
                  <button key={item.id} type="button" className="swatch" style={{ background: item.color }} title={item.name} aria-label={`Choose ${item.name} outfit`} aria-pressed={outfit.id === item.id} onClick={() => setOutfit(item)} data-testid={`button-outfit-${item.id}`} />
                ))}
                <span className="swatch-label">Outfit preset</span>
              </div>
              <div className="divider" />
              <div className="side-head" style={{ marginBottom: 4 }}><h2>Stage preferences</h2><SlidersHorizontal size={14} color="#858b7e" /></div>
              <div className="setting-row">
                <div className="setting-copy"><strong>Pose landmarks</strong><small>Show live joint points</small></div>
                <button type="button" className="switch" role="switch" aria-checked={showSkeleton} aria-label="Show pose landmarks" onClick={() => setShowSkeleton((value) => !value)} data-testid="switch-landmarks"><span /></button>
              </div>
              <div className="setting-row">
                <div className="setting-copy"><strong>Demo avatar preview</strong><small>Show the 3D figure on the stage</small></div>
                <button type="button" className="switch" role="switch" aria-checked={showDemoAvatar} aria-label="Show demo avatar overlay" onClick={() => setShowDemoAvatar((value) => !value)} data-testid="switch-avatar-overlay"><span /></button>
              </div>
              <div className="expression-panel" data-testid="panel-expression-cues">
                <div className="expression-heading">
                  <strong>Visible expression cues</strong>
                  <span>{displayedFaceCues ? `${displayedFaceCues.landmarkCount} points` : 'no face detected'}</span>
                </div>
                {emotionEstimate && (
                  <div className="expression-estimate-card" data-testid="active-emotion-estimate">
                    <span className="estimate-icon">{emotionEstimate.icon}</span>
                    <div className="estimate-text">
                      <div className="estimate-title">Inferred Expression: <b>{emotionEstimate.label}</b> ({Math.round(emotionEstimate.confidence * 100)}%)</div>
                      <small className="estimate-caption">Estimated from smile, brow, and eye blendshapes</small>
                    </div>
                  </div>
                )}
                {[
                  ['Smile cue', displayedFaceCues?.smile ?? 0],
                  ['Frown cue', displayedFaceCues?.frown ?? 0],
                  ['Surprise cue', displayedFaceCues?.surprise ?? 0],
                  ['Neutral cue', displayedFaceCues?.neutral ?? 0],
                ].map(([label, value]) => (
                  <div className="expression-row" key={label as string}>
                    <span>{label}</span>
                    <div className="expression-meter"><i style={{ width: `${Math.round(Number(value) * 100)}%` }} /></div>
                    <b>{displayedFaceCues ? `${Math.round(Number(value) * 100)}%` : '—'}</b>
                  </div>
                ))}
                <p>These are visible facial cues and expression estimates, not a determination of someone’s true emotional state.</p>
              </div>
              <div className="range-wrap">
                <div className="range-label"><span>Tracking response</span><b>{sensitivity}</b></div>
                <input type="range" min="20" max="100" value={sensitivity} onChange={(event) => setSensitivity(Number(event.target.value))} aria-label="Tracking response" data-testid="input-sensitivity" />
              </div>
            </section>

            <section className="panel side-panel">
              <div className="side-head">
                <h2>Motion sessions</h2>
                <span data-testid="status-session">
                  {recorder.phase === 'recording' ? 'RECORDING' :
                    recorder.phase === 'recording-paused' ? 'PAUSED' :
                      recorder.phase === 'playing' ? 'PLAYING' :
                        recorder.phase === 'playback-paused' ? 'PLAYBACK PAUSED' : 'READY'}
                </span>
              </div>
              <div className="session-card">
                <div className="session-icon">{isRecording ? <Video size={16} /> : recorder.recordings.length ? <Check size={16} /> : <Save size={16} />}</div>
                <div className="session-detail">
                  <strong>{recorder.activeRecordingName ?? recorder.recordings[0]?.name ?? 'No saved takes'}</strong>
                  <small>{isRecording ? 'Capturing motion locally' : recorder.recordings.length ? `${recorder.recordings.length} saved in this browser` : 'Start tracking, then record a take'}</small>
                </div>
                <span className="session-badge">
                  {isRecording ? recordTimeLabel : isPlayback ? playbackTimeLabel : 'LOCAL'}
                </span>
              </div>
              <div className="session-actions">
                {recorder.phase === 'recording' && (
                  <>
                    <button type="button" className="secondary-button" onClick={recorder.pauseRecording} data-testid="button-pause-recording"><Pause size={13} /> Pause</button>
                    <button type="button" className="primary-button" onClick={() => void recorder.stopRecording()} data-testid="button-stop-recording"><Square size={12} /> Save take</button>
                  </>
                )}
                {recorder.phase === 'recording-paused' && (
                  <>
                    <button type="button" className="secondary-button" onClick={recorder.resumeRecording} data-testid="button-resume-recording"><Play size={13} /> Resume</button>
                    <button type="button" className="primary-button" onClick={() => void recorder.stopRecording()} data-testid="button-stop-recording"><Square size={12} /> Save take</button>
                  </>
                )}
                {recorder.phase === 'playing' && (
                  <>
                    <button type="button" className="secondary-button" onClick={recorder.pausePlayback} data-testid="button-pause-playback"><Pause size={13} /> Pause</button>
                    <button type="button" className="secondary-button" onClick={recorder.restartPlayback} data-testid="button-restart-playback"><RotateCcw size={13} /> Restart</button>
                    <button type="button" className="secondary-button" onClick={recorder.stopPlayback} data-testid="button-stop-playback"><Square size={12} /> Stop</button>
                  </>
                )}
                {recorder.phase === 'playback-paused' && (
                  <>
                    <button type="button" className="primary-button" onClick={recorder.resumePlayback} data-testid="button-resume-playback"><Play size={13} /> Resume</button>
                    <button type="button" className="secondary-button" onClick={recorder.restartPlayback} data-testid="button-restart-playback"><RotateCcw size={13} /> Restart</button>
                    <button type="button" className="secondary-button" onClick={recorder.stopPlayback} data-testid="button-stop-playback"><Square size={12} /> Stop</button>
                  </>
                )}
                {recorder.phase === 'idle' && (
                  <>
                    <button type="button" className="primary-button" onClick={recorder.startRecording} disabled={status !== 'tracking'} data-testid="button-record-session"><Save size={13} /> Record a take</button>
                    {recorder.playbackRecording && (
                      <button type="button" className="secondary-button" onClick={recorder.restartPlayback} data-testid="button-replay-session"><RotateCcw size={13} /> Replay</button>
                    )}
                  </>
                )}
              </div>
              {recorder.playbackRecording && recorder.playbackDurationMs > 0 && (
                <div className="playback-timeline">
                  <div><span>{playbackTimeLabel}</span><span>{playbackDurationLabel}</span></div>
                  <input type="range" min={0} max={recorder.playbackDurationMs} value={recorder.playbackTimeMs} onChange={(event) => recorder.seekPlayback(Number(event.target.value))} aria-label="Playback timeline" data-testid="input-playback-timeline" />
                </div>
              )}
              <div className="privacy-note"><LockKeyhole size={13} /><span>Only timestamped motion and expression cues are stored in this browser. Webcam video is never recorded or uploaded.</span></div>
              <div className="recording-library">
                <div className="library-heading"><strong>Saved takes</strong><span>{recorder.recordings.length}</span></div>
                {recorder.recordings.length === 0 ? (
                  <div className="library-empty" data-testid="empty-recordings">No saved sessions yet.</div>
                ) : recorder.recordings.map((recording) => (
                  <div className="recording-row" key={recording.id} data-testid={`row-recording-${recording.id}`}>
                    <button type="button" className="recording-play" onClick={() => recorder.playRecording(recording)} aria-label={`Play ${recording.name}`} data-testid={`button-play-recording-${recording.id}`}><Play size={12} /></button>
                    <div className="recording-copy"><strong>{recording.name}</strong><small>{formatDuration(recording.durationMs)} · {new Date(recording.createdAt).toLocaleDateString()}</small></div>
                    <button type="button" className="recording-action" onClick={() => exportRecording(recording)} aria-label={`Export ${recording.name} as JSON`} data-testid={`button-export-recording-${recording.id}`}><Download size={13} /></button>
                    <button type="button" className="recording-action danger" onClick={() => void deleteRecording(recording)} aria-label={`Delete ${recording.name}`} data-testid={`button-delete-recording-${recording.id}`}><Trash2 size={13} /></button>
                  </div>
                ))}
              </div>
              {recorder.error && <div className="error-message" role="alert" data-testid="status-recording-error">{recorder.error}</div>}
            </section>
          </aside>
        </div>
      </main>
    </div>
  );
}

export default App;
