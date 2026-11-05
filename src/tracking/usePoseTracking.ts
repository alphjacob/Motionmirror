import { useCallback, useEffect, useRef, useState } from 'react';
import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
  type FaceLandmarkerResult,
  type HandLandmarkerResult,
  type PoseLandmarkerResult,
} from '@mediapipe/tasks-vision';
import type {
  EmotionEstimate,
  FaceCues,
  InferredEmotion,
  PoseCalibration,
  PoseLandmark,
  PoseTrackingState,
  TrackedHand,
  TrackingStatus,
} from './types';

const WASM_PATH =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1/wasm';
const POSE_MODEL_PATH =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
const HAND_MODEL_PATH =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const FACE_MODEL_PATH =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const MIN_TRACKING_CONFIDENCE = 0.4;
const AUXILIARY_TRACKING_INTERVAL_MS = 100;

type VisionFileset = Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>;

async function createPoseTask(vision: VisionFileset) {
  const options = {
    runningMode: 'VIDEO' as const,
    numPoses: 1,
    minPoseDetectionConfidence: MIN_TRACKING_CONFIDENCE,
    minPosePresenceConfidence: MIN_TRACKING_CONFIDENCE,
    minTrackingConfidence: MIN_TRACKING_CONFIDENCE,
  };
  try {
    return await PoseLandmarker.createFromOptions(vision, {
      ...options,
      baseOptions: { modelAssetPath: POSE_MODEL_PATH, delegate: 'GPU' },
    });
  } catch {
    return PoseLandmarker.createFromOptions(vision, {
      ...options,
      baseOptions: { modelAssetPath: POSE_MODEL_PATH, delegate: 'CPU' },
    });
  }
}

async function createHandTask(vision: VisionFileset) {
  const options = {
    runningMode: 'VIDEO' as const,
    numHands: 2,
    minHandDetectionConfidence: 0.45,
    minHandPresenceConfidence: 0.45,
    minTrackingConfidence: 0.45,
  };
  try {
    return await HandLandmarker.createFromOptions(vision, {
      ...options,
      baseOptions: { modelAssetPath: HAND_MODEL_PATH, delegate: 'GPU' },
    });
  } catch {
    return HandLandmarker.createFromOptions(vision, {
      ...options,
      baseOptions: { modelAssetPath: HAND_MODEL_PATH, delegate: 'CPU' },
    });
  }
}

async function createFaceTask(vision: VisionFileset) {
  const options = {
    runningMode: 'VIDEO' as const,
    numFaces: 1,
    minFaceDetectionConfidence: 0.45,
    minFacePresenceConfidence: 0.45,
    minTrackingConfidence: 0.45,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
  };
  try {
    return await FaceLandmarker.createFromOptions(vision, {
      ...options,
      baseOptions: { modelAssetPath: FACE_MODEL_PATH, delegate: 'GPU' },
    });
  } catch {
    return FaceLandmarker.createFromOptions(vision, {
      ...options,
      baseOptions: { modelAssetPath: FACE_MODEL_PATH, delegate: 'CPU' },
    });
  }
}

function averageConfidence(points: PoseLandmark[]) {
  const importantIndices = [0, 11, 12, 13, 14, 15, 16, 23, 24];
  const values = importantIndices
    .map((index) => points[index]?.visibility ?? points[index]?.presence)
    .filter((value): value is number => typeof value === 'number');

  if (!values.length) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function smoothLandmarks(
  previous: PoseLandmark[],
  next: PoseLandmark[],
  sensitivity: number,
): PoseLandmark[] {
  if (previous.length !== next.length) return next;
  const response = Math.min(0.8, Math.max(0.2, sensitivity / 100));
  return next.map((point, index) => {
    const old = previous[index];
    return {
      x: old.x + (point.x - old.x) * response,
      y: old.y + (point.y - old.y) * response,
      z: old.z + (point.z - old.z) * response,
      visibility: point.visibility,
      presence: point.presence,
    };
  });
}

function readHands(result: HandLandmarkerResult): TrackedHand[] {
  return result.landmarks.map((landmarks, index) => {
    const label = result.handedness[index]?.[0]?.categoryName;
    const handedness =
      label === 'Left' || label === 'Right' ? label : 'Unknown';
    return {
      handedness,
      landmarks: landmarks.map((point) => ({
        x: point.x,
        y: point.y,
        z: point.z,
        visibility: point.visibility,
        presence: point.presence,
      })),
      worldLandmarks: result.worldLandmarks?.[index]?.map((point) => ({
        x: point.x,
        y: point.y,
        z: point.z,
        visibility: point.visibility,
        presence: point.presence,
      })),
    };
  });
}

export function estimateEmotion(cues: {
  smile: number;
  frown: number;
  surprise: number;
  neutral: number;
}): EmotionEstimate {
  const { smile, frown, surprise, neutral } = cues;

  if (smile >= 0.32 && smile > frown + 0.10 && smile > surprise * 0.85) {
    return {
      emotion: 'Happy',
      confidence: Math.min(1, Math.max(0.4, smile)),
      label: 'Happy',
      icon: '😊',
    };
  }

  if (surprise >= 0.30 && surprise > smile * 0.9) {
    return {
      emotion: 'Surprised',
      confidence: Math.min(1, Math.max(0.4, surprise)),
      label: 'Surprised',
      icon: '😲',
    };
  }

  if (frown >= 0.28 && frown > smile + 0.08) {
    return {
      emotion: 'Sad',
      confidence: Math.min(1, Math.max(0.4, frown)),
      label: 'Sad',
      icon: '🙁',
    };
  }

  return {
    emotion: 'Neutral',
    confidence: Math.min(1, Math.max(0.5, neutral)),
    label: 'Neutral',
    icon: '😐',
  };
}

function readFaceCues(result: FaceLandmarkerResult, previousCues?: FaceCues | null): FaceCues | null {
  const face = result.faceLandmarks?.[0];
  if (!face?.length) {
    if (previousCues && (previousCues.smile > 0.04 || previousCues.frown > 0.04 || previousCues.surprise > 0.04)) {
      // Smooth decay toward neutral
      const decay = 0.65;
      const smile = previousCues.smile * decay;
      const frown = previousCues.frown * decay;
      const surprise = previousCues.surprise * decay;
      const jawOpen = previousCues.jawOpen * decay;
      const neutral = Math.max(0, 1 - Math.max(smile, frown, surprise));
      const emotion = estimateEmotion({ smile, frown, surprise, neutral });
      return {
        smile: Math.max(0, smile),
        frown: Math.max(0, frown),
        surprise: Math.max(0, surprise),
        neutral,
        jawOpen: Math.max(0, jawOpen),
        leftBlink: 0,
        rightBlink: 0,
        landmarkCount: previousCues.landmarkCount,
        emotion,
      };
    }
    return null;
  }

  const categories = result.faceBlendshapes?.[0]?.categories ?? [];
  const scores = new Map(
    categories.map((category) => [category.categoryName, category.score]),
  );
  const get = (name: string) => scores.get(name) ?? 0;
  const clamp = (value: number) => Math.max(0, Math.min(1, value));

  const rawSmile = (get('mouthSmileLeft') + get('mouthSmileRight')) / 2;
  const mouthFrown = (get('mouthFrownLeft') + get('mouthFrownRight')) / 2;
  const browDown = (get('browDownLeft') + get('browDownRight')) / 2;
  const rawFrown = Math.max(mouthFrown, mouthFrown * 0.6 + browDown * 0.4);

  const browUp = (get('browInnerUp') + (get('browOuterUpLeft') + get('browOuterUpRight')) * 0.5) * 0.5;
  const eyeWide = (get('eyeWideLeft') + get('eyeWideRight')) / 2;
  const jaw = get('jawOpen');
  const rawSurprise = Math.max(
    browUp * 0.55 + jaw * 0.45,
    (browUp + eyeWide + jaw) / 3,
  );

  const rawJawOpen = get('jawOpen');
  const rawLeftBlink = get('eyeBlinkLeft');
  const rawRightBlink = get('eyeBlinkRight');

  const alpha = 0.40;
  const smile = previousCues ? clamp(previousCues.smile + (rawSmile - previousCues.smile) * alpha) : clamp(rawSmile);
  const frown = previousCues ? clamp(previousCues.frown + (rawFrown - previousCues.frown) * alpha) : clamp(rawFrown);
  const surprise = previousCues ? clamp(previousCues.surprise + (rawSurprise - previousCues.surprise) * alpha) : clamp(rawSurprise);
  const jawOpen = previousCues ? clamp(previousCues.jawOpen + (rawJawOpen - previousCues.jawOpen) * alpha) : clamp(rawJawOpen);
  const leftBlink = clamp(rawLeftBlink);
  const rightBlink = clamp(rawRightBlink);

  const strongestCue = Math.max(smile, frown, surprise);
  const neutral = clamp(1 - strongestCue);
  const emotion = estimateEmotion({ smile, frown, surprise, neutral });

  return {
    smile,
    frown,
    surprise,
    neutral,
    jawOpen,
    leftBlink,
    rightBlink,
    landmarkCount: face.length,
    emotion,
  };
}

function readableCameraError(error: unknown) {
  if (!window.isSecureContext) {
    return 'Camera access requires a secure browser connection (HTTPS or localhost).';
  }
  if (error instanceof DOMException) {
    if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
      return 'Camera permission was denied. Allow camera access in your browser settings, then try again.';
    }
    if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
      return 'No camera was found. Connect a camera and try again.';
    }
    if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
      return 'The camera is already in use by another app or could not be started.';
    }
  }
  return error instanceof Error
    ? error.message
    : 'The camera or pose tracker could not be started.';
}

export function usePoseTracking(sensitivity = 68): PoseTrackingState {
  const videoRef = useRef<HTMLVideoElement>(null);
  const sensitivityRef = useRef(sensitivity);
  sensitivityRef.current = sensitivity;
  const landmarkerRef = useRef<PoseLandmarker | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const faceCuesRef = useRef<FaceCues | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const lastVideoTimeRef = useRef(-1);
  const lastAuxiliaryTimeRef = useRef(0);
  const landmarksRef = useRef<PoseLandmark[]>([]);
  const avatarLandmarksRef = useRef<PoseLandmark[]>([]);
  const lastGoodPoseAtRef = useRef(0);
  const recoveryUntilRef = useRef(0);
  const runningRef = useRef(false);
  const startingRef = useRef(false);
  const generationRef = useRef(0);
  const [status, setStatus] = useState<TrackingStatus>('idle');
  const [cameraReady, setCameraReady] = useState(false);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [landmarks, setLandmarks] = useState<PoseLandmark[]>([]);
  const [avatarLandmarks, setAvatarLandmarks] = useState<PoseLandmark[]>([]);
  const [calibration, setCalibration] = useState<PoseCalibration | null>(null);
  const [hands, setHands] = useState<TrackedHand[]>([]);
  const [faceCues, setFaceCues] = useState<FaceCues | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [featureWarning, setFeatureWarning] = useState<string | null>(null);

  const stopTracking = useCallback(() => {
    generationRef.current += 1;
    startingRef.current = false;
    runningRef.current = false;
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setCameraReady(false);

    const video = videoRef.current;
    if (video) {
      video.pause();
      video.srcObject = null;
    }

    landmarkerRef.current?.close();
    landmarkerRef.current = null;
    handLandmarkerRef.current?.close();
    handLandmarkerRef.current = null;
    faceLandmarkerRef.current?.close();
    faceLandmarkerRef.current = null;
    faceCuesRef.current = null;
    landmarksRef.current = [];
    avatarLandmarksRef.current = [];
    lastGoodPoseAtRef.current = 0;
    recoveryUntilRef.current = 0;
    setLandmarks([]);
    setAvatarLandmarks([]);
    setConfidence(null);
    setCalibration(null);
    setHands([]);
    setFaceCues(null);
    setError(null);
    setFeatureWarning(null);
    setStatus('idle');
  }, []);

  const startTracking = useCallback(async () => {
    if (runningRef.current || startingRef.current) return;
    startingRef.current = true;
    const generation = ++generationRef.current;
    setStatus('loading');
    setCameraReady(false);
    setError(null);

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('This browser does not support webcam access.');
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: 'user',
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30, max: 30 },
        },
      });
      if (generation !== generationRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      streamRef.current = stream;
      stream.getVideoTracks()[0]?.addEventListener(
        'ended',
        () => {
          if (generation !== generationRef.current) return;
          stopTracking();
          setError('The camera disconnected. Reconnect it and start tracking again.');
          setStatus('error');
        },
        { once: true },
      );

      const video = videoRef.current;
      if (!video) throw new Error('The camera preview is not ready yet.');
      video.srcObject = stream;
      await video.play();
      if (generation !== generationRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      setCameraReady(true);

      const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
      if (generation !== generationRef.current) return;
      const warnings: string[] = [];
      const [landmarker, handLandmarker, faceLandmarker] = await Promise.all([
        createPoseTask(vision),
        createHandTask(vision).catch(() => {
          warnings.push('Hand tracking is unavailable; body tracking will continue.');
          return null;
        }),
        createFaceTask(vision).catch(() => {
          warnings.push('Face cues are unavailable; body tracking will continue.');
          return null;
        }),
      ]);
      if (generation !== generationRef.current) {
        landmarker.close();
        handLandmarker?.close();
        faceLandmarker?.close();
        return;
      }
      landmarkerRef.current = landmarker;
      handLandmarkerRef.current = handLandmarker;
      faceLandmarkerRef.current = faceLandmarker;
      lastVideoTimeRef.current = -1;
      lastAuxiliaryTimeRef.current = 0;
      setFeatureWarning(warnings.length ? warnings.join(' ') : null);
      runningRef.current = true;
      startingRef.current = false;
      setStatus('lost');

      const processFrame = () => {
        if (!runningRef.current) return;
        const activeVideo = videoRef.current;
        const activeLandmarker = landmarkerRef.current;

        if (
          activeVideo &&
          activeLandmarker &&
          activeVideo.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
          activeVideo.currentTime !== lastVideoTimeRef.current
        ) {
          try {
            const timestamp = performance.now();
            lastVideoTimeRef.current = activeVideo.currentTime;
            const result: PoseLandmarkerResult =
              activeLandmarker.detectForVideo(activeVideo, timestamp);
            const detected = result.landmarks?.[0] as
              | PoseLandmark[]
              | undefined;
            const detectedWorld = result.worldLandmarks?.[0] as
              | PoseLandmark[]
              | undefined;

            if (detected?.length && detectedWorld?.length) {
              if (
                lastGoodPoseAtRef.current > 0 &&
                timestamp - lastGoodPoseAtRef.current > 250
              ) {
                recoveryUntilRef.current = timestamp + 300;
              }
              const response =
                timestamp < recoveryUntilRef.current
                  ? Math.min(sensitivityRef.current, 35)
                  : sensitivityRef.current;
              const filtered = smoothLandmarks(
                landmarksRef.current,
                detected,
                response,
              );
              const filteredWorld = smoothLandmarks(
                avatarLandmarksRef.current,
                detectedWorld,
                response,
              );
              const score = averageConfidence(filtered);
              landmarksRef.current = filtered;
              avatarLandmarksRef.current = filteredWorld;
              lastGoodPoseAtRef.current = timestamp;
              setLandmarks(filtered);
              setAvatarLandmarks(filteredWorld);
              setConfidence(score);
              setStatus(score === null || score >= MIN_TRACKING_CONFIDENCE ? 'tracking' : 'lost');
            } else {
              setConfidence(null);
              setStatus('lost');
            }

            if (timestamp - lastAuxiliaryTimeRef.current >= AUXILIARY_TRACKING_INTERVAL_MS) {
              lastAuxiliaryTimeRef.current = timestamp;
              const activeHands = handLandmarkerRef.current;
              const activeFace = faceLandmarkerRef.current;
              if (activeHands) {
                try {
                  setHands(readHands(activeHands.detectForVideo(activeVideo, timestamp)));
                } catch {
                  activeHands.close();
                  handLandmarkerRef.current = null;
                  setHands([]);
                  setFeatureWarning('Hand tracking stopped unexpectedly; body tracking is still active.');
                }
              }
              if (activeFace) {
                try {
                  const cues = readFaceCues(activeFace.detectForVideo(activeVideo, timestamp), faceCuesRef.current);
                  faceCuesRef.current = cues;
                  setFaceCues(cues);
                } catch {
                  activeFace.close();
                  faceLandmarkerRef.current = null;
                  faceCuesRef.current = null;
                  setFaceCues(null);
                  setFeatureWarning('Face tracking stopped unexpectedly; body tracking is still active.');
                }
              }
            }
          } catch (frameError) {
            runningRef.current = false;
            streamRef.current?.getTracks().forEach((track) => track.stop());
            streamRef.current = null;
            activeVideo.pause();
            activeVideo.srcObject = null;
            setCameraReady(false);
            activeLandmarker.close();
            landmarkerRef.current = null;
            handLandmarkerRef.current?.close();
            handLandmarkerRef.current = null;
            faceLandmarkerRef.current?.close();
            faceLandmarkerRef.current = null;
            setError(readableCameraError(frameError));
            setStatus('error');
            return;
          }
        }

        animationFrameRef.current = requestAnimationFrame(processFrame);
      };

      animationFrameRef.current = requestAnimationFrame(processFrame);
    } catch (startError) {
      if (generation !== generationRef.current) return;
      startingRef.current = false;
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      const video = videoRef.current;
      if (video) {
        video.pause();
        video.srcObject = null;
      }
      setCameraReady(false);
      landmarkerRef.current?.close();
      landmarkerRef.current = null;
      handLandmarkerRef.current?.close();
      handLandmarkerRef.current = null;
      faceLandmarkerRef.current?.close();
      faceLandmarkerRef.current = null;
      setError(readableCameraError(startError));
      setStatus('error');
    } finally {
      if (generation === generationRef.current) startingRef.current = false;
    }
  }, [stopTracking]);

  const calibrate = useCallback(() => {
    const currentWorld = avatarLandmarksRef.current;
    const left = currentWorld[11];
    const right = currentWorld[12];
    if (!left || !right) return;
    const shoulderWidth = Math.hypot(
      right.x - left.x,
      right.y - left.y,
      right.z - left.z,
    );
    if (shoulderWidth < 0.04) return;
    setCalibration({ shoulderWidth });
  }, []);

  useEffect(() => stopTracking, [stopTracking]);

  return {
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
  };
}
