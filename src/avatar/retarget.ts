import type { PoseCalibration, PoseLandmark } from '@/tracking/types';

export const DEFAULT_AVATAR_WORLD_SCALE = 1.5;
const TARGET_SHOULDER_WIDTH_METERS = 0.6;

export interface AvatarPoint {
  x: number;
  y: number;
  z: number;
}

export function mapLandmarkToCoverPreview(
  point: Pick<PoseLandmark, 'x' | 'y'>,
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number,
) {
  if (
    sourceWidth <= 0 ||
    sourceHeight <= 0 ||
    targetWidth <= 0 ||
    targetHeight <= 0
  ) {
    return { x: point.x, y: point.y };
  }

  const coverScale = Math.max(
    targetWidth / sourceWidth,
    targetHeight / sourceHeight,
  );
  const renderedWidth = sourceWidth * coverScale;
  const renderedHeight = sourceHeight * coverScale;
  const cropX = (renderedWidth - targetWidth) / 2;
  const cropY = (renderedHeight - targetHeight) / 2;

  return {
    x: (point.x * renderedWidth - cropX) / targetWidth,
    y: (point.y * renderedHeight - cropY) / targetHeight,
  };
}

/**
 * Convert MediaPipe's camera-space landmarks to a front-facing avatar space.
 * Camera x is inverted to match the mirrored selfie preview.
 */
export function getAvatarCalibrationScale(
  calibration: PoseCalibration | null | undefined,
) {
  if (!calibration || !Number.isFinite(calibration.shoulderWidth)) {
    return DEFAULT_AVATAR_WORLD_SCALE;
  }
  return Math.max(
    0.45,
    Math.min(
      3.5,
      TARGET_SHOULDER_WIDTH_METERS / calibration.shoulderWidth,
    ),
  );
}

export function mapLandmarkToAvatar(
  point: PoseLandmark,
  calibration: PoseCalibration | null | undefined,
  scale = getAvatarCalibrationScale(calibration),
): AvatarPoint {
  return {
    // MediaPipe Tasks-Vision worldLandmarks uses a y-DOWN convention (positive y =
    // below hips). Three.js uses y-UP. Negate y to convert between the two systems.
    // Also negate x so the avatar mirrors the camera image (selfie orientation).
    x: -point.x * scale,
    y: -point.y * scale,
    z: Math.max(-0.8, Math.min(0.8, point.z * scale)),
  };
}

/** Convert old v1 recordings, which stored normalized image coordinates, to
 * equivalent world-space coordinates so they remain playable after retargeting.
 * Image-space y=0 is the top of the frame; y=1 is the bottom.
 * We convert to MediaPipe-style world coords (y-DOWN, origin at hips) so that
 * mapLandmarkToAvatar (which negates y) will produce the correct upright pose.
 */
export function legacyImagePointToWorld(point: PoseLandmark): PoseLandmark {
  const inverseScale = 3.1 / DEFAULT_AVATAR_WORLD_SCALE;
  return {
    ...point,
    x: (point.x - 0.5) * inverseScale,
    // Image y=0 is top (head), y=1 is bottom (feet). Hips ~y=0.55 in image space.
    // Convert so that positive y = below hips (MediaPipe y-DOWN convention).
    // mapLandmarkToAvatar will then negate y to get Three.js y-UP.
    y: (point.y - 0.5) * inverseScale,
    z: (-point.z * 1.1) / DEFAULT_AVATAR_WORLD_SCALE,
  };
}
