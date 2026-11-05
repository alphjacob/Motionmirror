import type { AvatarOutfit } from '@/avatar/DemoAvatar';
import type {
  FaceCues,
  PoseLandmark,
  TrackedHand,
} from '@/tracking/types';

export interface MotionFrame {
  timeMs: number;
  /** Normalized image-space landmarks retained for export/debug overlays. */
  landmarks: PoseLandmark[];
  /** MediaPipe world-space pose landmarks in meters; absent on legacy v1 takes. */
  worldLandmarks?: PoseLandmark[];
  hands: TrackedHand[];
  faceCues: FaceCues | null;
  confidence: number | null;
}

export interface MotionRecording {
  version: 1 | 2;
  id: string;
  name: string;
  createdAt: string;
  durationMs: number;
  outfit: AvatarOutfit;
  sampleRate: number;
  frames: MotionFrame[];
}

export type RecorderPhase =
  | 'idle'
  | 'recording'
  | 'recording-paused'
  | 'playing'
  | 'playback-paused';
