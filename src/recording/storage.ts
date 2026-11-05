import type { MotionRecording } from './types';

const DATABASE_NAME = 'motionmirror-local';
const DATABASE_VERSION = 1;
const STORE_NAME = 'recordings';

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function isPosePoint(value: unknown) {
  if (!value || typeof value !== 'object') return false;
  const point = value as Record<string, unknown>;
  return (
    isFiniteNumber(point.x) &&
    isFiniteNumber(point.y) &&
    isFiniteNumber(point.z)
  );
}

function isHand(value: unknown) {
  if (!value || typeof value !== 'object') return false;
  const hand = value as Record<string, unknown>;
  return (
    (hand.handedness === 'Left' ||
      hand.handedness === 'Right' ||
      hand.handedness === 'Unknown') &&
    Array.isArray(hand.landmarks) &&
    hand.landmarks.every(isPosePoint) &&
    (hand.worldLandmarks === undefined ||
      (Array.isArray(hand.worldLandmarks) &&
        hand.worldLandmarks.every(isPosePoint)))
  );
}

function isFaceCues(value: unknown) {
  if (!value || typeof value !== 'object') return false;
  const cues = value as Record<string, unknown>;
  return (
    isFiniteNumber(cues.smile) &&
    isFiniteNumber(cues.frown) &&
    isFiniteNumber(cues.surprise) &&
    isFiniteNumber(cues.neutral) &&
    isFiniteNumber(cues.jawOpen) &&
    isFiniteNumber(cues.leftBlink) &&
    isFiniteNumber(cues.rightBlink) &&
    isFiniteNumber(cues.landmarkCount)
  );
}

function isMotionRecording(value: unknown): value is MotionRecording {
  if (!value || typeof value !== 'object') return false;
  const recording = value as Record<string, unknown>;
  if (
    (recording.version !== 1 && recording.version !== 2) ||
    typeof recording.id !== 'string' ||
    typeof recording.name !== 'string' ||
    typeof recording.createdAt !== 'string' ||
    !Number.isFinite(Date.parse(recording.createdAt)) ||
    !isFiniteNumber(recording.durationMs) ||
    recording.durationMs < 0 ||
    !isFiniteNumber(recording.sampleRate) ||
    recording.sampleRate <= 0 ||
    !['mocap', 'casual', 'future'].includes(String(recording.outfit)) ||
    !Array.isArray(recording.frames) ||
    recording.frames.length === 0 ||
    recording.frames.length > 200000
  ) {
    return false;
  }

  let lastTime = -1;
  return recording.frames.every((value) => {
    if (!value || typeof value !== 'object') return false;
    const frame = value as Record<string, unknown>;
    const valid =
      isFiniteNumber(frame.timeMs) &&
      frame.timeMs >= 0 &&
      frame.timeMs >= lastTime &&
      Array.isArray(frame.landmarks) &&
      frame.landmarks.every(isPosePoint) &&
      (frame.worldLandmarks === undefined ||
        (Array.isArray(frame.worldLandmarks) &&
          frame.worldLandmarks.every(isPosePoint))) &&
      (recording.version !== 2 ||
        (Array.isArray(frame.worldLandmarks) &&
          frame.worldLandmarks.every(isPosePoint))) &&
      Array.isArray(frame.hands) &&
      frame.hands.every(isHand) &&
      (frame.faceCues === null || isFaceCues(frame.faceCues)) &&
      (frame.confidence === null || isFiniteNumber(frame.confidence));
    if (valid) lastTime = frame.timeMs as number;
    return valid;
  });
}

function openDatabase() {
  if (!('indexedDB' in window)) {
    return Promise.reject(
      new Error('This browser does not support local recording storage.'),
    );
  }

  return new Promise<IDBDatabase>((resolve, reject) => {
    const request = window.indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(STORE_NAME)) {
        database.createObjectStore(STORE_NAME, { keyPath: 'id' });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () =>
      reject(request.error ?? new Error('Could not open local recording storage.'));
    request.onblocked = () =>
      reject(new Error('Close other MotionMirror tabs to update local recording storage.'));
  });
}

async function runRequest<T>(
  mode: IDBTransactionMode,
  operation: (store: IDBObjectStore) => IDBRequest<T>,
) {
  const database = await openDatabase();
  return new Promise<T>((resolve, reject) => {
    const transaction = database.transaction(STORE_NAME, mode);
    const request = operation(transaction.objectStore(STORE_NAME));
    let result: T;
    request.onsuccess = () => {
      result = request.result;
    };
    request.onerror = () =>
      reject(request.error ?? new Error('The local recording operation failed.'));
    transaction.oncomplete = () => {
      database.close();
      resolve(result);
    };
    transaction.onerror = () => {
      database.close();
      reject(transaction.error ?? new Error('The local recording transaction failed.'));
    };
    transaction.onabort = () => {
      database.close();
      reject(transaction.error ?? new Error('The local recording transaction was cancelled.'));
    };
  });
}

export async function saveMotionRecording(recording: MotionRecording) {
  if (!isMotionRecording(recording)) {
    throw new Error('The recording is incomplete or has an unsupported format.');
  }
  await runRequest('readwrite', (store) => store.put(recording));
}

export async function listMotionRecordings(): Promise<MotionRecording[]> {
  const values = await runRequest<unknown[]>('readonly', (store) => store.getAll());
  if (!values.every(isMotionRecording)) {
    throw new Error('A saved recording is invalid or uses an unsupported format.');
  }
  return values.sort(
    (left, right) =>
      new Date(right.createdAt).getTime() - new Date(left.createdAt).getTime(),
  );
}

export async function deleteMotionRecording(id: string) {
  if (!id.trim()) throw new Error('A recording identifier is required.');
  await runRequest('readwrite', (store) => store.delete(id));
}
