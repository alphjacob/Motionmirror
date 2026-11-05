import type { RefObject } from 'react';

export type TrackingStatus = 'idle' | 'loading' | 'tracking' | 'lost' | 'error';

export interface PoseLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
  presence?: number;
}

export interface PoseCalibration {
  /** Shoulder separation in MediaPipe world meters. */
  shoulderWidth: number;
}

export interface TrackedHand {
  handedness: 'Left' | 'Right' | 'Unknown';
  landmarks: PoseLandmark[];
  worldLandmarks?: PoseLandmark[];
}

export type InferredEmotion = 'Neutral' | 'Happy' | 'Sad' | 'Surprised';

export interface EmotionEstimate {
  emotion: InferredEmotion;
  confidence: number;
  label: string;
  icon: string;
}

export interface FaceCues {
  smile: number;
  frown: number;
  surprise: number;
  neutral: number;
  jawOpen: number;
  leftBlink: number;
  rightBlink: number;
  landmarkCount: number;
  emotion?: EmotionEstimate;
}

export interface PoseTrackingState {
  videoRef: RefObject<HTMLVideoElement | null>;
  cameraReady: boolean;
  status: TrackingStatus;
  confidence: number | null;
  /** Normalized image-space landmarks for the camera overlay. */
  landmarks: PoseLandmark[];
  /** MediaPipe world-space landmarks in meters for avatar retargeting. */
  avatarLandmarks: PoseLandmark[];
  hands: TrackedHand[];
  faceCues: FaceCues | null;
  calibration: PoseCalibration | null;
  error: string | null;
  featureWarning: string | null;
  startTracking: () => Promise<void>;
  stopTracking: () => void;
  calibrate: () => void;
}
