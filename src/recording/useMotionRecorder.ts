import { useCallback, useEffect, useRef, useState } from 'react';
import type { AvatarOutfit } from '@/avatar/DemoAvatar';
import type {
  FaceCues,
  PoseLandmark,
  TrackedHand,
} from '@/tracking/types';
import {
  deleteMotionRecording,
  listMotionRecordings,
  saveMotionRecording,
} from './storage';
import type {
  MotionFrame,
  MotionRecording,
  RecorderPhase,
} from './types';

const SAMPLE_INTERVAL_MS = 100;
const PLAYBACK_REFRESH_MS = 40;

interface RecorderInput {
  landmarks: PoseLandmark[];
  avatarLandmarks: PoseLandmark[];
  hands: TrackedHand[];
  faceCues: FaceCues | null;
  confidence: number | null;
  tracking: boolean;
  outfit: AvatarOutfit;
}

export interface MotionRecorderState {
  phase: RecorderPhase;
  elapsedMs: number;
  recordings: MotionRecording[];
  playbackFrame: MotionFrame | null;
  playbackRecording: MotionRecording | null;
  playbackTimeMs: number;
  playbackDurationMs: number;
  activeRecordingName: string | null;
  error: string | null;
  refreshRecordings: () => Promise<void>;
  startRecording: () => void;
  pauseRecording: () => void;
  resumeRecording: () => void;
  stopRecording: () => Promise<void>;
  playRecording: (recording: MotionRecording) => void;
  pausePlayback: () => void;
  resumePlayback: () => void;
  restartPlayback: () => void;
  stopPlayback: () => void;
  seekPlayback: (timeMs: number) => void;
  deleteRecording: (id: string) => Promise<void>;
}

function interpolateNumber(from: number, to: number, amount: number) {
  return from + (to - from) * amount;
}

function interpolateFrame(recording: MotionRecording, timeMs: number): MotionFrame | null {
  if (!recording.frames.length) return null;
  let afterIndex = recording.frames.findIndex((frame) => frame.timeMs >= timeMs);
  if (afterIndex < 0) afterIndex = recording.frames.length - 1;
  const after = recording.frames[afterIndex];
  const before = recording.frames[Math.max(0, afterIndex - (after.timeMs > timeMs ? 1 : 0))];
  const span = after.timeMs - before.timeMs;
  const amount = span > 0 ? Math.max(0, Math.min(1, (timeMs - before.timeMs) / span)) : 0;

  const landmarks = before.landmarks.map((point, index) => {
    const next = after.landmarks[index] ?? point;
    return {
      x: interpolateNumber(point.x, next.x, amount),
      y: interpolateNumber(point.y, next.y, amount),
      z: interpolateNumber(point.z, next.z, amount),
      visibility: point.visibility,
      presence: point.presence,
    };
  });
  const worldLandmarks = before.worldLandmarks?.map((point, index) => {
    const next = after.worldLandmarks?.[index] ?? point;
    return {
      x: interpolateNumber(point.x, next.x, amount),
      y: interpolateNumber(point.y, next.y, amount),
      z: interpolateNumber(point.z, next.z, amount),
      visibility: point.visibility,
      presence: point.presence,
    };
  });
  const hands = before.hands.map((hand) => {
    const nextHand = after.hands.find((candidate) => candidate.handedness === hand.handedness);
    const worldLandmarks = hand.worldLandmarks?.map((point, index) => {
      const next = nextHand?.worldLandmarks?.[index] ?? point;
      return {
        x: interpolateNumber(point.x, next.x, amount),
        y: interpolateNumber(point.y, next.y, amount),
        z: interpolateNumber(point.z, next.z, amount),
        visibility: point.visibility,
        presence: point.presence,
      };
    });
    return {
      handedness: hand.handedness,
      landmarks: hand.landmarks.map((point, index) => {
        const next = nextHand?.landmarks[index] ?? point;
        return {
          x: interpolateNumber(point.x, next.x, amount),
          y: interpolateNumber(point.y, next.y, amount),
          z: interpolateNumber(point.z, next.z, amount),
          visibility: point.visibility,
          presence: point.presence,
        };
      }),
      ...(worldLandmarks ? { worldLandmarks } : {}),
    };
  });
  let faceCues: FaceCues | null = before.faceCues;
  if (before.faceCues && after.faceCues) {
    faceCues = {
      smile: interpolateNumber(before.faceCues.smile, after.faceCues.smile, amount),
      frown: interpolateNumber(before.faceCues.frown, after.faceCues.frown, amount),
      surprise: interpolateNumber(before.faceCues.surprise, after.faceCues.surprise, amount),
      neutral: interpolateNumber(before.faceCues.neutral, after.faceCues.neutral, amount),
      jawOpen: interpolateNumber(before.faceCues.jawOpen, after.faceCues.jawOpen, amount),
      leftBlink: interpolateNumber(before.faceCues.leftBlink, after.faceCues.leftBlink, amount),
      rightBlink: interpolateNumber(before.faceCues.rightBlink, after.faceCues.rightBlink, amount),
      landmarkCount: before.faceCues.landmarkCount,
    };
  }
  return {
    timeMs,
    landmarks,
    ...(worldLandmarks ? { worldLandmarks } : {}),
    hands,
    faceCues,
    confidence:
      before.confidence === null || after.confidence === null
        ? before.confidence
        : interpolateNumber(before.confidence, after.confidence, amount),
  };
}

export function useMotionRecorder(input: RecorderInput): MotionRecorderState {
  const inputRef = useRef(input);
  inputRef.current = input;
  const phaseRef = useRef<RecorderPhase>('idle');
  const framesRef = useRef<MotionFrame[]>([]);
  const metadataRef = useRef<Pick<
    MotionRecording,
    'id' | 'name' | 'createdAt' | 'outfit'
  > | null>(null);
  const captureIntervalRef = useRef<number | null>(null);
  const startTimeRef = useRef(0);
  const pauseStartedRef = useRef(0);
  const pausedDurationRef = useRef(0);
  const playbackIntervalRef = useRef<number | null>(null);
  const playbackSessionRef = useRef<MotionRecording | null>(null);
  const playbackStartTimeRef = useRef(0);
  const playbackStartPositionRef = useRef(0);
  const playbackTimeRef = useRef(0);

  const [phase, setPhaseState] = useState<RecorderPhase>('idle');
  const [elapsedMs, setElapsedMs] = useState(0);
  const [recordings, setRecordings] = useState<MotionRecording[]>([]);
  const [playbackFrame, setPlaybackFrame] = useState<MotionFrame | null>(null);
  const [playbackRecording, setPlaybackRecording] = useState<MotionRecording | null>(null);
  const [playbackTimeMs, setPlaybackTimeMs] = useState(0);
  const [playbackDurationMs, setPlaybackDurationMs] = useState(0);
  const [activeRecordingName, setActiveRecordingName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const setPhase = useCallback((next: RecorderPhase) => {
    phaseRef.current = next;
    setPhaseState(next);
  }, []);

  const refreshRecordings = useCallback(async () => {
    try {
      setRecordings(await listMotionRecordings());
      setError(null);
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : 'Could not load saved recordings.');
    }
  }, []);

  const captureFrame = useCallback(() => {
    const current = inputRef.current;
    if (
      !current.tracking ||
      current.landmarks.length === 0 ||
      current.avatarLandmarks.length === 0
    ) return;
    const timeMs = Math.max(
      0,
      Date.now() - startTimeRef.current - pausedDurationRef.current,
    );
    const lastFrame = framesRef.current.at(-1);
    if (lastFrame && timeMs <= lastFrame.timeMs) return;
    framesRef.current.push({
      timeMs,
      landmarks: current.landmarks.map((point) => ({ ...point })),
      worldLandmarks: current.avatarLandmarks.map((point) => ({ ...point })),
      hands: current.hands.map((hand) => ({
        handedness: hand.handedness,
        landmarks: hand.landmarks.map((point) => ({ ...point })),
        ...(hand.worldLandmarks
          ? { worldLandmarks: hand.worldLandmarks.map((point) => ({ ...point })) }
          : {}),
      })),
      faceCues: current.faceCues ? { ...current.faceCues } : null,
      confidence: current.confidence,
    });
    setElapsedMs(timeMs);
  }, []);

  const startCaptureInterval = useCallback(() => {
    if (captureIntervalRef.current !== null) {
      window.clearInterval(captureIntervalRef.current);
    }
    captureIntervalRef.current = window.setInterval(captureFrame, SAMPLE_INTERVAL_MS);
    captureFrame();
  }, [captureFrame]);

  const clearCaptureInterval = useCallback(() => {
    if (captureIntervalRef.current !== null) {
      window.clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
  }, []);

  const clearPlaybackInterval = useCallback(() => {
    if (playbackIntervalRef.current !== null) {
      window.clearInterval(playbackIntervalRef.current);
      playbackIntervalRef.current = null;
    }
  }, []);

  const startRecording = useCallback(() => {
    if (!inputRef.current.tracking) {
      setError('Start camera tracking before recording a motion session.');
      return;
    }
    if (
      phaseRef.current === 'recording' ||
      phaseRef.current === 'recording-paused'
    ) {
      return;
    }
    if (!globalThis.crypto?.randomUUID) {
      setError('This browser cannot create a secure recording identifier.');
      return;
    }
    clearPlaybackInterval();
    playbackSessionRef.current = null;
    framesRef.current = [];
    startTimeRef.current = Date.now();
    pauseStartedRef.current = 0;
    pausedDurationRef.current = 0;
    const createdAt = new Date().toISOString();
    metadataRef.current = {
      id: crypto.randomUUID(),
      name: `Take · ${new Date(createdAt).toLocaleString()}`,
      createdAt,
      outfit: inputRef.current.outfit,
    };
    setElapsedMs(0);
    setActiveRecordingName(metadataRef.current.name);
    setPlaybackRecording(null);
    setPlaybackFrame(null);
    setPlaybackTimeMs(0);
    setPlaybackDurationMs(0);
    setError(null);
    setPhase('recording');
    startCaptureInterval();
  }, [clearCaptureInterval, clearPlaybackInterval, setPhase, startCaptureInterval]);

  const pauseRecording = useCallback(() => {
    if (phaseRef.current !== 'recording') return;
    clearCaptureInterval();
    pauseStartedRef.current = Date.now();
    setPhase('recording-paused');
  }, [clearCaptureInterval, setPhase]);

  const resumeRecording = useCallback(() => {
    if (phaseRef.current !== 'recording-paused') return;
    pausedDurationRef.current += Date.now() - pauseStartedRef.current;
    pauseStartedRef.current = 0;
    setPhase('recording');
    startCaptureInterval();
  }, [setPhase, startCaptureInterval]);

  const stopRecording = useCallback(async () => {
    if (
      phaseRef.current !== 'recording' &&
      phaseRef.current !== 'recording-paused'
    ) return;
    clearCaptureInterval();
    if (pauseStartedRef.current) {
      pausedDurationRef.current += Date.now() - pauseStartedRef.current;
      pauseStartedRef.current = 0;
    }
    captureFrame();
    const metadata = metadataRef.current;
    const frames = framesRef.current;
    setPhase('idle');
    if (!metadata || frames.length === 0) {
      setActiveRecordingName(null);
      setError('No tracked pose frames were captured. Keep a person visible while recording.');
      return;
    }
    const durationMs = Math.max(
      frames.at(-1)?.timeMs ?? 0,
      Date.now() - startTimeRef.current - pausedDurationRef.current,
    );
    const recording: MotionRecording = {
      version: 2,
      ...metadata,
      durationMs,
      sampleRate: 1000 / SAMPLE_INTERVAL_MS,
      frames,
    };
    try {
      await saveMotionRecording(recording);
      await refreshRecordings();
      setActiveRecordingName(null);
      setError(null);
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : 'Could not save this recording.');
    }
  }, [captureFrame, clearCaptureInterval, refreshRecordings, setPhase]);

  const updatePlaybackFrame = useCallback((timeMs: number) => {
    const recording = playbackSessionRef.current;
    if (!recording) return;
    const clamped = Math.max(0, Math.min(recording.durationMs, timeMs));
    playbackTimeRef.current = clamped;
    setPlaybackTimeMs(clamped);
    setPlaybackFrame(interpolateFrame(recording, clamped));
  }, []);

  const startPlaybackInterval = useCallback(() => {
    clearPlaybackInterval();
    playbackStartPositionRef.current = playbackTimeRef.current;
    playbackStartTimeRef.current = performance.now();
    playbackIntervalRef.current = window.setInterval(() => {
      const recording = playbackSessionRef.current;
      if (!recording) return;
      const timeMs =
        playbackStartPositionRef.current +
        (performance.now() - playbackStartTimeRef.current);
      updatePlaybackFrame(timeMs);
      if (timeMs >= recording.durationMs) {
        clearPlaybackInterval();
        setPhase('idle');
      }
    }, PLAYBACK_REFRESH_MS);
    setPhase('playing');
  }, [clearPlaybackInterval, setPhase, updatePlaybackFrame]);

  const playRecording = useCallback((recording: MotionRecording) => {
    if (
      phaseRef.current === 'recording' ||
      phaseRef.current === 'recording-paused'
    ) {
      setError('Stop the current recording before starting playback.');
      return;
    }
    clearPlaybackInterval();
    playbackSessionRef.current = recording;
    setPlaybackRecording(recording);
    playbackTimeRef.current = 0;
    setPlaybackDurationMs(recording.durationMs);
    setActiveRecordingName(recording.name);
    setError(null);
    setPlaybackFrame(interpolateFrame(recording, 0));
    startPlaybackInterval();
  }, [clearPlaybackInterval, startPlaybackInterval]);

  const pausePlayback = useCallback(() => {
    if (phaseRef.current !== 'playing') return;
    const position =
      playbackStartPositionRef.current +
      (performance.now() - playbackStartTimeRef.current);
    updatePlaybackFrame(position);
    clearPlaybackInterval();
    setPhase('playback-paused');
  }, [clearPlaybackInterval, setPhase, updatePlaybackFrame]);

  const resumePlayback = useCallback(() => {
    if (phaseRef.current !== 'playback-paused' || !playbackSessionRef.current) return;
    startPlaybackInterval();
  }, [startPlaybackInterval]);

  const restartPlayback = useCallback(() => {
    if (!playbackSessionRef.current) return;
    playbackTimeRef.current = 0;
    updatePlaybackFrame(0);
    startPlaybackInterval();
  }, [startPlaybackInterval, updatePlaybackFrame]);

  const stopPlayback = useCallback(() => {
    clearPlaybackInterval();
    playbackSessionRef.current = null;
    setPlaybackRecording(null);
    playbackTimeRef.current = 0;
    setPlaybackFrame(null);
    setPlaybackTimeMs(0);
    setPlaybackDurationMs(0);
    setActiveRecordingName(null);
    if (
      phaseRef.current === 'playing' ||
      phaseRef.current === 'playback-paused'
    ) {
      setPhase('idle');
    }
  }, [clearPlaybackInterval, setPhase]);

  const seekPlayback = useCallback((timeMs: number) => {
    updatePlaybackFrame(timeMs);
    if (phaseRef.current === 'playing') {
      playbackStartPositionRef.current = playbackTimeRef.current;
      playbackStartTimeRef.current = performance.now();
    }
  }, [updatePlaybackFrame]);

  const deleteRecording = useCallback(async (id: string) => {
    try {
      await deleteMotionRecording(id);
      if (playbackSessionRef.current?.id === id) stopPlayback();
      await refreshRecordings();
      setError(null);
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : 'Could not delete this recording.');
    }
  }, [refreshRecordings, stopPlayback]);

  useEffect(() => {
    void refreshRecordings();
    return () => {
      clearCaptureInterval();
      clearPlaybackInterval();
    };
  }, [clearCaptureInterval, clearPlaybackInterval, refreshRecordings]);

  return {
    phase,
    elapsedMs,
    recordings,
    playbackFrame,
    playbackRecording,
    playbackTimeMs,
    playbackDurationMs,
    activeRecordingName,
    error,
    refreshRecordings,
    startRecording,
    pauseRecording,
    resumeRecording,
    stopRecording,
    playRecording,
    pausePlayback,
    resumePlayback,
    restartPlayback,
    stopPlayback,
    seekPlayback,
    deleteRecording,
  };
}
