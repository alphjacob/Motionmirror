import { Canvas } from '@react-three/fiber';
import { useMemo, useRef, useState } from 'react';
import { BufferGeometry, Color, Float32BufferAttribute, Quaternion, Vector3 } from 'three';
import type {
  FaceCues,
  PoseCalibration,
  PoseLandmark,
  TrackedHand,
} from '@/tracking/types';
import {
  getAvatarCalibrationScale,
  legacyImagePointToWorld,
  mapLandmarkToAvatar,
} from './retarget';

export type AvatarOutfit = 'mocap' | 'casual' | 'future';

interface DemoAvatarProps {
  landmarks: PoseLandmark[];
  calibration?: PoseCalibration | null;
  hands: TrackedHand[];
  faceCues: FaceCues | null;
  outfit: AvatarOutfit;
  active: boolean;
  transparent?: boolean;
}

export interface OutfitStyle {
  skin: string;
  hair: string;
  top: string;
  topTrim: string;
  bottom: string;
  shoes: string;
  accent: string;
  roughness: number;
  metalness: number;
  emissive?: string;
  emissiveIntensity?: number;
}

export const OUTFIT_STYLES: Record<AvatarOutfit, OutfitStyle> = {
  casual: {
    skin: '#dca482',
    hair: '#362b24',
    top: '#3b74db',       // Classic royal/indigo blue T-shirt
    topTrim: '#2554ad',   // Collar trim
    bottom: '#25313d',    // Dark slate denim trousers
    shoes: '#f8fafc',     // Clean white sneakers with dark rubber soles
    accent: '#f59e42',    // Subtle warm accent
    roughness: 0.65,
    metalness: 0.05,
  },
  mocap: {
    skin: '#dca482',
    hair: '#23292e',
    top: '#1e252b',       // Sleek dark charcoal compression suit
    topTrim: '#151a1e',
    bottom: '#181e24',    // Athletic tights
    shoes: '#12161a',     // Mocap tracking booties
    accent: '#3ef09e',    // High-visibility phosphor mint tracking markers
    roughness: 0.55,
    metalness: 0.18,
    emissive: '#0d3d23',
    emissiveIntensity: 0.25,
  },
  future: {
    skin: '#dca482',
    hair: '#141e28',
    top: '#0e1c27',       // Cyber chassis
    topTrim: '#00f2d5',
    bottom: '#09131c',    // Cyber exosuit legs
    shoes: '#081017',     // Cyber armor sabatons
    accent: '#00f2d5',    // Glowing cyan cyber power conduits
    roughness: 0.28,
    metalness: 0.68,
    emissive: '#00e5ff',
    emissiveIntensity: 0.65,
  },
};

export const OUTFIT_COLORS: Record<AvatarOutfit, { suit: string; accent: string; skin: string }> = {
  mocap:  { suit: OUTFIT_STYLES.mocap.top,  accent: OUTFIT_STYLES.mocap.accent,  skin: OUTFIT_STYLES.mocap.skin },
  casual: { suit: OUTFIT_STYLES.casual.top, accent: OUTFIT_STYLES.casual.accent, skin: OUTFIT_STYLES.casual.skin },
  future: { suit: OUTFIT_STYLES.future.top, accent: OUTFIT_STYLES.future.accent, skin: OUTFIT_STYLES.future.skin },
};

// MediaPipe worldLandmarks standard coordinates:
// Origin: Center between hips.
// Positive X = Person's Left (screen left in mirror view).
// Negative X = Person's Right (screen right in mirror view).
// Negative Y = Upwards (towards head).
// Positive Y = Downwards (towards feet).
// Positive Z = Forward (towards camera).
const NEUTRAL: Record<number, PoseLandmark> = {
  0:  { x:  0.00, y: -0.91, z:  0.03 }, // nose
  7:  { x:  0.08, y: -0.88, z: -0.04 }, // left ear
  8:  { x: -0.08, y: -0.88, z: -0.04 }, // right ear
  11: { x:  0.20, y: -0.54, z:  0.00 }, // left shoulder
  12: { x: -0.20, y: -0.54, z:  0.00 }, // right shoulder
  13: { x:  0.31, y: -0.31, z:  0.00 }, // left elbow
  14: { x: -0.31, y: -0.31, z:  0.00 }, // right elbow
  15: { x:  0.37, y: -0.08, z:  0.00 }, // left wrist
  16: { x: -0.37, y: -0.08, z:  0.00 }, // right wrist
  23: { x:  0.10, y:  0.00, z:  0.00 }, // left hip
  24: { x: -0.10, y:  0.00, z:  0.00 }, // right hip
  25: { x:  0.11, y:  0.48, z:  0.00 }, // left knee
  26: { x: -0.11, y:  0.48, z:  0.00 }, // right knee
  27: { x:  0.11, y:  0.91, z:  0.00 }, // left ankle
  28: { x: -0.11, y:  0.91, z:  0.00 }, // right ankle
  31: { x:  0.14, y:  0.98, z:  0.10 }, // left foot
  32: { x: -0.14, y:  0.98, z:  0.10 }, // right foot
};

function pointToWorld(
  point: PoseLandmark,
  calibration: PoseCalibration | null | undefined,
  scale: number,
): Vector3 {
  const mapped = mapLandmarkToAvatar(point, calibration, scale);
  return new Vector3(mapped.x, mapped.y, mapped.z);
}

/**
 * Validates a landmark's confidence and coordinates. If occluded or missing,
 * returns the reliable fallback position instead of snapping the body into a distorted pose.
 */
function getValidLandmark(
  index: number,
  landmarks: PoseLandmark[],
  fallback: PoseLandmark,
  minConfidence = 0.35,
): PoseLandmark {
  const lm = landmarks[index];
  if (!lm) return fallback;
  const score = lm.visibility ?? lm.presence;
  if (typeof score === 'number' && score < minConfidence) {
    return fallback;
  }
  if (!Number.isFinite(lm.x) || !Number.isFinite(lm.y) || !Number.isFinite(lm.z)) {
    return fallback;
  }
  return lm;
}

/**
 * Tapered bone segment connecting precisely between its parent and child joints.
 * Supports distinct proximal (radiusTop) and distal (radiusBottom) radii for natural human limb tapering.
 */
function Bone({
  start,
  end,
  radius,
  radiusTop,
  radiusBottom,
  color,
  roughness = 0.45,
  metalness = 0.10,
  emissive,
  emissiveIntensity = 0,
}: {
  start: Vector3;
  end: Vector3;
  radius?: number;
  radiusTop?: number;
  radiusBottom?: number;
  color: string;
  roughness?: number;
  metalness?: number;
  emissive?: string;
  emissiveIntensity?: number;
}) {
  const rTop = radiusTop ?? radius ?? 0.05;
  const rBottom = radiusBottom ?? radius ?? 0.05;
  const direction = new Vector3().subVectors(end, start);
  const length = direction.length();
  if (length < 0.0001) return null;
  const position = new Vector3().addVectors(start, end).multiplyScalar(0.5);
  const rotation = new Quaternion().setFromUnitVectors(
    new Vector3(0, 1, 0),
    direction.normalize(),
  );

  return (
    <group>
      {/* Tapered cylinder spanning start to end */}
      <mesh position={position} quaternion={rotation}>
        <cylinderGeometry args={[rBottom, rTop, length, 16, 1, false]} />
        <meshStandardMaterial
          color={color}
          roughness={roughness}
          metalness={metalness}
          emissive={emissive}
          emissiveIntensity={emissiveIntensity}
        />
      </mesh>
      {/* Seamless spherical joint caps at both ends */}
      <mesh position={start}>
        <sphereGeometry args={[rTop, 16, 12]} />
        <meshStandardMaterial
          color={color}
          roughness={roughness}
          metalness={metalness}
          emissive={emissive}
          emissiveIntensity={emissiveIntensity}
        />
      </mesh>
      <mesh position={end}>
        <sphereGeometry args={[rBottom, 16, 12]} />
        <meshStandardMaterial
          color={color}
          roughness={roughness}
          metalness={metalness}
          emissive={emissive}
          emissiveIntensity={emissiveIntensity}
        />
      </mesh>
    </group>
  );
}

function Joint({
  point,
  radius,
  color,
  roughness = 0.35,
  metalness = 0.15,
  emissive,
  emissiveIntensity = 0,
}: {
  point: Vector3;
  radius: number;
  color: string;
  roughness?: number;
  metalness?: number;
  emissive?: string;
  emissiveIntensity?: number;
}) {
  return (
    <mesh position={point}>
      <sphereGeometry args={[radius, 18, 14]} />
      <meshStandardMaterial
        color={color}
        roughness={roughness}
        metalness={metalness}
        emissive={emissive}
        emissiveIntensity={emissiveIntensity}
      />
    </mesh>
  );
}

/**
 * Procedural shoe / sneaker figure bridging the ankle joint to the forefoot.
 * Creates an intentional sneaker shape with rubber sole base.
 */
function FootFigure({
  ankle,
  toe,
  style,
  outfit,
  shoulderWidth,
}: {
  ankle: Vector3;
  toe: Vector3;
  style: OutfitStyle;
  outfit: AvatarOutfit;
  shoulderWidth: number;
}) {
  const direction = new Vector3().subVectors(toe, ankle);
  const length = Math.max(direction.length(), shoulderWidth * 0.28);
  const mid = new Vector3().addVectors(ankle, toe).multiplyScalar(0.5);
  const solePos = new Vector3(mid.x, mid.y - shoulderWidth * 0.038, mid.z);
  const rotation = new Quaternion().setFromUnitVectors(
    new Vector3(0, 0, 1),
    new Vector3(direction.x, 0, direction.z || 0.001).normalize(),
  );

  const shoeWidth = shoulderWidth * 0.098;
  const shoeHeight = shoulderWidth * 0.072;

  return (
    <group>
      {/* Foot bridge from ankle to toe */}
      <Bone
        start={ankle}
        end={toe}
        radiusTop={shoulderWidth * 0.058}
        radiusBottom={shoulderWidth * 0.045}
        color={outfit === 'casual' ? style.shoes : style.top}
        roughness={style.roughness}
        metalness={style.metalness}
      />
      {/* Rubber outsole */}
      <mesh position={solePos} quaternion={rotation}>
        <boxGeometry args={[shoeWidth, shoeHeight * 0.35, length * 1.08]} />
        <meshStandardMaterial
          color={outfit === 'casual' ? '#1e293b' : '#0e141a'}
          roughness={0.82}
        />
      </mesh>
      {/* Shoe upper */}
      <mesh
        position={new Vector3(solePos.x, solePos.y + shoeHeight * 0.38, solePos.z)}
        quaternion={rotation}
      >
        <boxGeometry args={[shoeWidth * 0.92, shoeHeight * 0.65, length * 0.96]} />
        <meshStandardMaterial
          color={style.shoes}
          roughness={style.roughness}
          metalness={style.metalness}
          emissive={outfit === 'future' ? style.accent : undefined}
          emissiveIntensity={outfit === 'future' ? 0.3 : 0}
        />
      </mesh>
    </group>
  );
}

const FINGER_CHAINS = [
  [0, 1, 2, 3, 4],    // thumb
  [0, 5, 6, 7, 8],    // index
  [0, 9, 10, 11, 12], // middle
  [0, 13, 14, 15, 16],// ring
  [0, 17, 18, 19, 20],// pinky
];

function HandFigure({
  hand,
  avatarLandmarks,
  calibration,
  scale,
  color,
}: {
  hand: TrackedHand;
  avatarLandmarks: PoseLandmark[];
  calibration: PoseCalibration | null | undefined;
  scale: number;
  color: string;
}) {
  const wristIndex = hand.handedness === 'Left' ? 15 : 16;
  const poseWrist = avatarLandmarks[wristIndex] ?? NEUTRAL[wristIndex];
  const handOrigin = hand.worldLandmarks?.[0];
  const points = useMemo(
    () => {
      if (hand.worldLandmarks?.length && handOrigin) {
        return hand.worldLandmarks.map((point) =>
          pointToWorld({
            ...point,
            x: poseWrist.x + point.x - handOrigin.x,
            y: poseWrist.y + point.y - handOrigin.y,
            // In MediaPipe worldLandmarks, -z points toward the camera (depth closer to lens).
            // In Three.js, the camera sits at +z looking at origin, so +z points toward the viewer.
            // Inverting the relative hand z-displacement ensures that palm vs back orientation
            // and forward/backward hand tilting match the real hand facing the webcam.
            z: poseWrist.z - (point.z - handOrigin.z),
          }, calibration, scale),
        );
      }
      return hand.landmarks.map((point) =>
        pointToWorld(legacyImagePointToWorld(point), null, scale),
      );
    },
    [
      hand.worldLandmarks,
      hand.landmarks,
      handOrigin,
      poseWrist,
      calibration,
      scale,
    ],
  );

  return (
    <group>
      {FINGER_CHAINS.flatMap((chain) =>
        chain.slice(0, -1).map((startIndex, segmentIndex) => {
          const start = points[startIndex];
          const end = points[chain[segmentIndex + 1]];
          if (!start || !end) return null;
          return (
            <Bone
              key={`${chain[0]}-${segmentIndex}`}
              start={start}
              end={end}
              radiusTop={segmentIndex === 0 ? 0.018 : 0.014}
              radiusBottom={segmentIndex === 0 ? 0.015 : 0.012}
              color={color}
            />
          );
        }),
      )}
      {[0, 4, 8, 12, 16, 20].map((index) =>
        points[index] ? (
          <Joint key={index} point={points[index]} radius={0.016} color={color} />
        ) : null,
      )}
    </group>
  );
}

/**
 * Humanoid procedural face and head:
 * - Natural cranial ellipse with defined jaw/chin contour.
 * - Stylized hairstyle with forehead fringe and sideburns.
 * - Anatomical ears, eyebrows, eyelids (with live blink tracking).
 * - Defined nose bridge/tip and animated lips (smile, frown, jawOpen).
 * - Outfit head accessories (mocap headband with tracker dot, cyber visor).
 */
function AvatarHead({
  point,
  tilt,
  cues,
  style,
  outfit,
  radius,
}: {
  point: Vector3;
  tilt: number;
  cues: FaceCues | null;
  style: OutfitStyle;
  outfit: AvatarOutfit;
  radius: number;
}) {
  const leftBlink = cues?.leftBlink ?? 0;
  const rightBlink = cues?.rightBlink ?? 0;
  const smile = cues?.smile ?? 0;
  const frown = cues?.frown ?? 0;
  const surprise = cues?.surprise ?? 0;
  const jawOpen = cues?.jawOpen ?? 0;

  // Mouth dynamics:
  // Smile curves corners upward (+Y) and widens mouth (+/-X).
  // Frown curves corners downward (-Y) and pulls corners slightly inward.
  // Surprise drops jaw and opens mouth into a rounded "O".
  const cornerX = radius * (0.16 + smile * 0.085 - frown * 0.02 - surprise * 0.035);
  const cornerY = -radius * 0.39 + smile * radius * 0.085 - frown * radius * 0.075 - jawOpen * radius * 0.08;
  const cornerZ = radius * 0.89;

  const upperCenterY = -radius * 0.36 + frown * radius * 0.015 - surprise * radius * 0.035;
  const upperCenterZ = radius * 0.93;

  const lowerCenterY = -radius * 0.42 - jawOpen * radius * 0.18 - surprise * radius * 0.09;
  const lowerCenterZ = radius * 0.915;

  const leftCorner = new Vector3(-cornerX, cornerY, cornerZ);
  const rightCorner = new Vector3(cornerX, cornerY, cornerZ);
  const upperCenter = new Vector3(0, upperCenterY, upperCenterZ);
  const lowerCenter = new Vector3(0, lowerCenterY, lowerCenterZ);

  // Eye open scale: widen on surprise, subtle crinkle on smile
  const leftEyeOpen = Math.max(0.10, (1 - leftBlink * 0.88) * (1 - smile * 0.12) * (1 + surprise * 0.28));
  const rightEyeOpen = Math.max(0.10, (1 - rightBlink * 0.88) * (1 - smile * 0.12) * (1 + surprise * 0.28));

  return (
    <group position={point} rotation={[0, 0, tilt]}>
      {/* 1. Cranium: Naturally proportioned human cranial ellipsoid */}
      <mesh scale={[0.92, 1.06, 0.96]}>
        <sphereGeometry args={[radius, 28, 22]} />
        <meshStandardMaterial color={style.skin} roughness={0.52} />
      </mesh>

      {/* 2. Jaw & Chin contour (moves subtly with jawOpen) */}
      <mesh
        position={[0, -radius * 0.52 - jawOpen * radius * 0.06, radius * 0.22]}
        scale={[0.46, 0.35, 0.42]}
      >
        <sphereGeometry args={[radius, 16, 12]} />
        <meshStandardMaterial color={style.skin} roughness={0.54} />
      </mesh>

      {/* 3. Stylized Modern Hairstyle */}
      <group>
        {/* Crown & Back volume */}
        <mesh position={[0, radius * 0.18, -radius * 0.08]} scale={[0.96, 0.94, 0.98]}>
          <sphereGeometry args={[radius, 24, 18]} />
          <meshStandardMaterial color={style.hair} roughness={0.78} />
        </mesh>
        {/* Forehead fringe */}
        <mesh position={[0, radius * 0.68, radius * 0.52]} scale={[0.76, 0.26, 0.42]}>
          <sphereGeometry args={[radius, 16, 12]} />
          <meshStandardMaterial color={style.hair} roughness={0.78} />
        </mesh>
        {/* Left & Right sideburns */}
        {([-1, 1] as const).map((side) => (
          <mesh
            key={`sideburn-${side}`}
            position={[side * radius * 0.84, -radius * 0.06, radius * 0.06]}
            scale={[0.12, 0.28, 0.20]}
          >
            <sphereGeometry args={[radius, 12, 10]} />
            <meshStandardMaterial color={style.hair} roughness={0.78} />
          </mesh>
        ))}
      </group>

      {/* 4. Anatomical Ears */}
      {([-1, 1] as const).map((side) => (
        <group
          key={`ear-${side}`}
          position={[side * radius * 0.88, -radius * 0.04, -radius * 0.06]}
          rotation={[0.08, -side * 0.24, side * 0.05]}
          scale={[radius * 0.14, radius * 0.32, radius * 0.22]}
        >
          <mesh>
            <sphereGeometry args={[1, 14, 12]} />
            <meshStandardMaterial color={style.skin} roughness={0.56} />
          </mesh>
        </group>
      ))}

      {/* 5. Eyebrows with emotional cue response */}
      {([-1, 1] as const).map((side) => {
        const browLift = surprise * radius * 0.13 - frown * radius * 0.06 + smile * radius * 0.02;
        // Furrow inner brows downward on frown; arch upward on surprise
        const browTilt = -side * (0.05 + frown * 0.30 - surprise * 0.14);
        return (
          <mesh
            key={`eyebrow-${side}`}
            position={[
              side * radius * 0.35,
              radius * 0.27 + browLift,
              radius * 0.90,
            ]}
            rotation={[0, 0, browTilt]}
            scale={[radius * 0.22, radius * 0.042, radius * 0.045]}
          >
            <sphereGeometry args={[1, 12, 8]} />
            <meshStandardMaterial color={style.hair} roughness={0.8} />
          </mesh>
        );
      })}

      {/* 6. Eyes: Sclera, Iris, Pupil, Catchlight, and dynamic Eyelid blink */}
      {([-1, 1] as const).map((side) => {
        const eyeOpen = side < 0 ? leftEyeOpen : rightEyeOpen;
        const blink = side < 0 ? leftBlink : rightBlink;
        return (
          <group key={`eye-${side}`} position={[side * radius * 0.35, radius * 0.12, radius * 0.88]}>
            {/* White Sclera */}
            <mesh scale={[1, eyeOpen, 0.5]}>
              <sphereGeometry args={[radius * 0.15, 16, 12]} />
              <meshStandardMaterial color="#fcfaf4" roughness={0.28} />
            </mesh>
            {/* Iris */}
            <mesh
              position={[0, 0, radius * 0.048]}
              scale={[1, eyeOpen, 0.35]}
            >
              <sphereGeometry args={[radius * 0.082, 14, 10]} />
              <meshStandardMaterial color="#3a6888" roughness={0.3} />
            </mesh>
            {/* Pupil */}
            <mesh
              position={[0, 0, radius * 0.065]}
              scale={[1, eyeOpen, 0.35]}
            >
              <sphereGeometry args={[radius * 0.046, 12, 8]} />
              <meshStandardMaterial color="#141816" roughness={0.25} />
            </mesh>
            {/* Specular Catchlight dot */}
            {blink < 0.75 && (
              <mesh position={[radius * 0.024, radius * 0.024, radius * 0.082]}>
                <sphereGeometry args={[radius * 0.016, 8, 6]} />
                <meshBasicMaterial color="#ffffff" />
              </mesh>
            )}
          </group>
        );
      })}

      {/* 7. Nose: Defined bridge and rounded tip */}
      <group>
        {/* Bridge */}
        <mesh position={[0, radius * 0.03, radius * 0.93]} scale={[0.07, 0.17, 0.11]}>
          <sphereGeometry args={[radius, 12, 8]} />
          <meshStandardMaterial color={style.skin} roughness={0.54} />
        </mesh>
        {/* Tip */}
        <mesh position={[0, -radius * 0.09, radius * 0.96]} scale={[0.13, 0.12, 0.14]}>
          <sphereGeometry args={[radius, 14, 10]} />
          <meshStandardMaterial color={style.skin} roughness={0.52} />
        </mesh>
      </group>

      {/* 8. Expressive 3D Mouth & Lips with Curved Corner Kinematics */}
      <group>
        {/* Upper Lip: Left & Right segments connecting center to corners */}
        <Bone
          start={leftCorner}
          end={upperCenter}
          radius={radius * 0.030}
          color="#884f50"
          roughness={0.46}
        />
        <Bone
          start={upperCenter}
          end={rightCorner}
          radius={radius * 0.030}
          color="#884f50"
          roughness={0.46}
        />
        {/* Upper Central Cupid's Bow fullness */}
        <mesh position={upperCenter}>
          <sphereGeometry args={[radius * 0.038, 12, 10]} />
          <meshStandardMaterial color="#884f50" roughness={0.46} />
        </mesh>

        {/* Lower Lip: Left & Right segments connecting center to corners */}
        <Bone
          start={leftCorner}
          end={lowerCenter}
          radius={radius * 0.032}
          color="#7a4445"
          roughness={0.46}
        />
        <Bone
          start={lowerCenter}
          end={rightCorner}
          radius={radius * 0.032}
          color="#7a4445"
          roughness={0.46}
        />
        {/* Lower Central Lip Pad fullness */}
        <mesh position={lowerCenter}>
          <sphereGeometry args={[radius * 0.044, 12, 10]} />
          <meshStandardMaterial color="#7a4445" roughness={0.46} />
        </mesh>

        {/* Mouth Corner Commissures */}
        <mesh position={leftCorner}>
          <sphereGeometry args={[radius * 0.028, 10, 8]} />
          <meshStandardMaterial color="#754345" roughness={0.5} />
        </mesh>
        <mesh position={rightCorner}>
          <sphereGeometry args={[radius * 0.028, 10, 8]} />
          <meshStandardMaterial color="#754345" roughness={0.5} />
        </mesh>

        {/* Inner Mouth Cavity (open when smiling, jaw dropped, or surprised) */}
        {(jawOpen > 0.06 || surprise > 0.15 || smile > 0.32) && (
          <mesh
            position={[0, (upperCenterY + lowerCenterY) * 0.5, radius * 0.87]}
            scale={[cornerX * 0.90, Math.max(0.005, (upperCenterY - lowerCenterY) * 0.52), radius * 0.05]}
          >
            <sphereGeometry args={[1, 12, 10]} />
            <meshStandardMaterial color="#220b0e" roughness={0.8} />
          </mesh>
        )}

        {/* Teeth row visible on broad smile or open mouth */}
        {(smile > 0.32 || jawOpen > 0.16) && (
          <mesh
            position={[0, upperCenterY - radius * 0.024, radius * 0.885]}
            scale={[cornerX * 0.70, radius * 0.022, radius * 0.028]}
          >
            <boxGeometry args={[1, 1, 1]} />
            <meshStandardMaterial color="#f8fafc" roughness={0.25} />
          </mesh>
        )}
      </group>

      {/* 9. Outfit-specific Head Accessories */}
      {outfit === 'mocap' && (
        <group>
          {/* Mocap Headband */}
          <mesh position={[0, radius * 0.24, 0]}>
            <cylinderGeometry args={[radius * 0.96, radius * 0.96, radius * 0.12, 24, 1, true]} />
            <meshStandardMaterial color="#1a2026" roughness={0.6} />
          </mesh>
          {/* Forehead tracking marker dot */}
          <mesh position={[0, radius * 0.24, radius * 0.98]}>
            <sphereGeometry args={[radius * 0.062, 12, 10]} />
            <meshStandardMaterial
              color={style.accent}
              emissive={style.accent}
              emissiveIntensity={0.5}
            />
          </mesh>
        </group>
      )}

      {outfit === 'future' && (
        <group>
          {/* Cyber Visor strip across upper face */}
          <mesh position={[0, radius * 0.12, radius * 0.82]} scale={[radius * 0.92, radius * 0.15, radius * 0.22]}>
            <boxGeometry args={[1, 1, 1]} />
            <meshStandardMaterial
              color="#00f2d5"
              emissive="#00f2d5"
              emissiveIntensity={0.75}
              roughness={0.2}
              metalness={0.8}
            />
          </mesh>
        </group>
      )}
    </group>
  );
}

/**
 * Procedural humanoid torso mesh generator:
 * Creates an anatomically contoured, seamless torso with natural chest pectorals/ribcage,
 * a narrower waist, and contoured pelvic girdle without stacked spheres or visible gaps.
 */
function createAnatomicalTorsoGeometry(
  tMin: number,
  tMax: number,
  shoulderWidth: number,
  hipWidth: number,
  torsoLength: number,
  numSlices = 18,
  numRadial = 24,
): BufferGeometry {
  const positions: number[] = [];
  const uvs: number[] = [];
  const indices: number[] = [];

  const getProfile = (t: number) => {
    let rx: number;
    let rz: number;
    let zOffset = 0;

    if (t < 0.28) {
      // Pelvis & Lower Hips (tMin to 0.28)
      // Normalizing t between -0.06 and 0.28 -> s in [0, 1]
      const s = Math.max(0, Math.min(1, (t - (-0.06)) / (0.28 - (-0.06))));
      // Anatomical flare at the iliac crest around s = 0.38 (t ~ 0.07)
      const crest = Math.sin(s * Math.PI) * 0.045;
      rx = (hipWidth * 0.44 * (1 - s) + shoulderWidth * 0.26 * s) + hipWidth * crest;
      rz = (hipWidth * 0.31 * (1 - s) + shoulderWidth * 0.185 * s) + hipWidth * crest * 0.4;
    } else if (t < 0.58) {
      // Abdomen & Narrow Natural Waist (0.28 to 0.58)
      const p = (t - 0.28) / (0.58 - 0.28);
      // Gentle inward waist cinch around p ~ 0.4 (t ~ 0.40)
      const indent = Math.sin(p * Math.PI) * 0.028;
      rx = shoulderWidth * (0.26 * (1 - p) + 0.35 * p - indent);
      rz = shoulderWidth * (0.185 * (1 - p) + 0.23 * p - indent * 0.6);
    } else {
      // Ribcage, Pectorals & Clavicular Shelf (0.58 to 1.0)
      const p = (t - 0.58) / (1.0 - 0.58);
      // Pectoral prominence around p ~ 0.55 (mid chest)
      const pec = Math.sin(p * Math.PI) * 0.042;
      // Clavicular broadening towards shoulders at top (p -> 1.0)
      rx = shoulderWidth * (0.35 * (1 - p) + 0.33 * p + pec);
      rz = shoulderWidth * (0.23 * (1 - p) + 0.19 * p + pec * 0.7);
      zOffset = pec * shoulderWidth * 0.30; // Anterior pectoral projection
    }

    return { rx, rz, zOffset };
  };

  for (let s = 0; s <= numSlices; s++) {
    const frac = s / numSlices;
    const t = tMin + (tMax - tMin) * frac;
    const y = (t - 0.5) * torsoLength;
    const { rx, rz, zOffset } = getProfile(t);

    for (let r = 0; r <= numRadial; r++) {
      const theta = (r / numRadial) * Math.PI * 2;
      const vx = rx * Math.cos(theta);
      const vy = y;
      const vz = zOffset + rz * Math.sin(theta);

      positions.push(vx, vy, vz);
      uvs.push(r / numRadial, frac);
    }
  }

  // Generate quad indices
  const ringSize = numRadial + 1;
  for (let s = 0; s < numSlices; s++) {
    for (let r = 0; r < numRadial; r++) {
      const a = s * ringSize + r;
      const b = (s + 1) * ringSize + r;
      const c = (s + 1) * ringSize + (r + 1);
      const d = s * ringSize + (r + 1);

      indices.push(a, b, d);
      indices.push(b, c, d);
    }
  }

  // Bottom cap
  const bottomCenterIdx = positions.length / 3;
  const bottomY = (tMin - 0.5) * torsoLength;
  const bottomProfile = getProfile(tMin);
  positions.push(0, bottomY, bottomProfile.zOffset);
  uvs.push(0.5, 0);
  for (let r = 0; r < numRadial; r++) {
    indices.push(bottomCenterIdx, r + 1, r);
  }

  // Top cap
  const topCenterIdx = positions.length / 3;
  const topY = (tMax - 0.5) * torsoLength;
  const topProfile = getProfile(tMax);
  positions.push(0, topY, topProfile.zOffset);
  uvs.push(0.5, 1);
  const topRingStart = numSlices * ringSize;
  for (let r = 0; r < numRadial; r++) {
    indices.push(topCenterIdx, topRingStart + r, topRingStart + r + 1);
  }

  const geom = new BufferGeometry();
  geom.setAttribute('position', new Float32BufferAttribute(positions, 3));
  geom.setAttribute('uv', new Float32BufferAttribute(uvs, 2));
  geom.setIndex(indices);
  geom.computeVertexNormals();
  return geom;
}

function AvatarFigure({
  landmarks,
  calibration,
  hands,
  faceCues,
  outfit,
  active,
}: DemoAvatarProps) {
  const style = OUTFIT_STYLES[outfit];
  const scale = getAvatarCalibrationScale(calibration);

  // Raw world landmarks with occluded-limb stabilization
  const rawPoints = useMemo(() => {
    // 1. Shoulders & Head
    const p11 = getValidLandmark(11, landmarks, NEUTRAL[11]);
    const p12 = getValidLandmark(12, landmarks, NEUTRAL[12]);
    const p0  = getValidLandmark(0,  landmarks, NEUTRAL[0]);
    const p7  = getValidLandmark(7,  landmarks, NEUTRAL[7]);
    const p8  = getValidLandmark(8,  landmarks, NEUTRAL[8]);

    // 2. Arms (fallback to resting pose hanging from shoulder if occluded)
    const p13Fallback: PoseLandmark = { x: p11.x + 0.11, y: p11.y + 0.23, z: p11.z };
    const p14Fallback: PoseLandmark = { x: p12.x - 0.11, y: p12.y + 0.23, z: p12.z };
    const p13 = getValidLandmark(13, landmarks, p13Fallback);
    const p14 = getValidLandmark(14, landmarks, p14Fallback);

    const p15Fallback: PoseLandmark = { x: p13.x + 0.06, y: p13.y + 0.23, z: p13.z };
    const p16Fallback: PoseLandmark = { x: p14.x - 0.06, y: p14.y + 0.23, z: p14.z };
    const p15 = getValidLandmark(15, landmarks, p15Fallback);
    const p16 = getValidLandmark(16, landmarks, p16Fallback);

    // 3. Hips (if seated / occluded, anchor hips below shoulders along spine)
    const shoulderMidX = (p11.x + p12.x) * 0.5;
    const shoulderMidY = (p11.y + p12.y) * 0.5;
    const shoulderMidZ = (p11.z + p12.z) * 0.5;
    const p23Fallback: PoseLandmark = { x: shoulderMidX + 0.10, y: shoulderMidY + 0.54, z: shoulderMidZ };
    const p24Fallback: PoseLandmark = { x: shoulderMidX - 0.10, y: shoulderMidY + 0.54, z: shoulderMidZ };
    const p23 = getValidLandmark(23, landmarks, p23Fallback);
    const p24 = getValidLandmark(24, landmarks, p24Fallback);

    // 4. Legs (if knees/ankles are occluded under desk, anchor naturally below hips)
    const p25Fallback: PoseLandmark = { x: p23.x + 0.01, y: p23.y + 0.48, z: p23.z };
    const p26Fallback: PoseLandmark = { x: p24.x - 0.01, y: p24.y + 0.48, z: p24.z };
    const p25 = getValidLandmark(25, landmarks, p25Fallback);
    const p26 = getValidLandmark(26, landmarks, p26Fallback);

    const p27Fallback: PoseLandmark = { x: p25.x, y: p25.y + 0.43, z: p25.z };
    const p28Fallback: PoseLandmark = { x: p26.x, y: p26.y + 0.43, z: p26.z };
    const p27 = getValidLandmark(27, landmarks, p27Fallback);
    const p28 = getValidLandmark(28, landmarks, p28Fallback);

    const p31Fallback: PoseLandmark = { x: p27.x + 0.03, y: p27.y + 0.07, z: p27.z + 0.10 };
    const p32Fallback: PoseLandmark = { x: p28.x - 0.03, y: p28.y + 0.07, z: p28.z + 0.10 };
    const p31 = getValidLandmark(31, landmarks, p31Fallback);
    const p32 = getValidLandmark(32, landmarks, p32Fallback);

    const activeLandmarks: Record<number, PoseLandmark> = {
      0: p0, 7: p7, 8: p8,
      11: p11, 12: p12, 13: p13, 14: p14, 15: p15, 16: p16,
      23: p23, 24: p24, 25: p25, 26: p26, 27: p27, 28: p28, 31: p31, 32: p32,
    };

    return Object.fromEntries(
      Object.entries(activeLandmarks).map(([idx, pt]) => [
        Number(idx),
        pointToWorld(pt, calibration, scale),
      ]),
    ) as Record<number, Vector3>;
  }, [landmarks, calibration, scale]);

  // Modest EMA smoothing filter (alpha = 0.60):
  // Eliminates camera sensor micro-jitter without adding perceptible latency.
  // Resets instantly if distance jump > 0.45m (e.g. calibration reset or playback seek).
  const smoothedPointsRef = useRef<Record<number, Vector3>>({});
  const points = useMemo(() => {
    const current = smoothedPointsRef.current;
    const result: Record<number, Vector3> = {};

    for (const [key, target] of Object.entries(rawPoints)) {
      const idx = Number(key);
      const prev = current[idx];
      if (!prev || prev.distanceTo(target) > 0.45) {
        current[idx] = target.clone();
      } else {
        current[idx] = prev.clone().lerp(target, 0.60);
      }
      result[idx] = current[idx];
    }
    return result;
  }, [rawPoints]);

  // Torso calculations & anatomical transforms
  const torsoTop = new Vector3()
    .addVectors(points[11], points[12])
    .multiplyScalar(0.5);
  const torsoBottom = new Vector3()
    .addVectors(points[23], points[24])
    .multiplyScalar(0.5);
  const torsoAxis = new Vector3().subVectors(torsoTop, torsoBottom);
  const torsoLength = Math.max(torsoAxis.length(), 0.28);
  const torsoRotation = new Quaternion().setFromUnitVectors(
    new Vector3(0, 1, 0),
    torsoAxis.clone().normalize(),
  );

  const shoulderWidth = Math.max(points[11].distanceTo(points[12]), 0.35);
  const hipWidth = Math.max(points[23].distanceTo(points[24]), 0.20);
  const torsoMidpoint = new Vector3().addVectors(torsoBottom, torsoTop).multiplyScalar(0.5);

  const { upperTorsoGeometry, lowerTorsoGeometry } = useMemo(() => {
    return {
      upperTorsoGeometry: createAnatomicalTorsoGeometry(0.28, 1.0, shoulderWidth, hipWidth, torsoLength),
      lowerTorsoGeometry: createAnatomicalTorsoGeometry(-0.06, 0.28, shoulderWidth, hipWidth, torsoLength),
    };
  }, [shoulderWidth, hipWidth, torsoLength]);
  const headRadius = Math.max(0.12, Math.min(0.20, shoulderWidth * 0.18));

  // Head center & tilt
  const eyeLeft = points[7];
  const eyeRight = points[8];
  const headTilt = Math.atan2(
    eyeRight.y - eyeLeft.y,
    eyeRight.x - eyeLeft.x,
  );

  const headCenter = new Vector3(
    points[0].x,
    points[0].y + headRadius * 0.14,
    points[0].z - headRadius * 0.28,
  );

  // Clothing color mappings per outfit
  const topColor = style.top;
  const bottomColor = style.bottom;
  const upperArmColor = outfit === 'casual' ? style.top : style.top;
  const forearmColor = outfit === 'casual' ? style.skin : (outfit === 'future' ? style.accent : style.top);
  const handColor = outfit === 'casual' ? style.skin : style.accent;

  return (
    <group>
      {/* 1. Structural Central Spine Core */}
      <Bone
        start={torsoBottom}
        end={torsoTop}
        radius={shoulderWidth * 0.055}
        color={topColor}
        roughness={style.roughness}
        metalness={style.metalness}
      />

      {/* 2. Anatomical Humanoid Torso (Seamless Upper Chest & Lower Pelvis Loft) */}
      <group position={torsoMidpoint} quaternion={torsoRotation}>
        {/* Upper Torso: Contoured Pectorals, Ribcage, and Inward Natural Waist Taper */}
        <mesh geometry={upperTorsoGeometry}>
          <meshStandardMaterial
            color={topColor}
            roughness={style.roughness}
            metalness={style.metalness}
          />
        </mesh>

        {/* Lower Torso: Contoured Pelvic Girdle seamlessly matching Waist and Hips */}
        <mesh geometry={lowerTorsoGeometry}>
          <meshStandardMaterial
            color={bottomColor}
            roughness={style.roughness}
            metalness={style.metalness}
          />
        </mesh>

        {/* Casual Outfit Waistband / Belt Trim at the natural waist line */}
        {outfit === 'casual' && (
          <mesh
            position={[0, (0.28 - 0.5) * torsoLength, 0]}
            scale={[shoulderWidth * 0.266, torsoLength * 0.038, shoulderWidth * 0.196]}
          >
            <cylinderGeometry args={[1, 1, 1, 24]} />
            <meshStandardMaterial color="#1a232c" roughness={0.7} />
          </mesh>
        )}

        {/* Futuristic Chest Core Reactor */}
        {outfit === 'future' && (
          <mesh
            position={[0, (0.72 - 0.5) * torsoLength, shoulderWidth * 0.14]}
            scale={[shoulderWidth * 0.08, shoulderWidth * 0.08, shoulderWidth * 0.08]}
          >
            <sphereGeometry args={[1, 14, 12]} />
            <meshStandardMaterial
              color={style.accent}
              emissive={style.accent}
              emissiveIntensity={0.8}
            />
          </mesh>
        )}
      </group>

      {/* 3. Clavicle Bar & Deltoid Shoulder Caps */}
      <Bone
        start={points[11]}
        end={points[12]}
        radius={shoulderWidth * 0.075}
        color={topColor}
        roughness={style.roughness}
        metalness={style.metalness}
      />
      {/* Left Deltoid cap */}
      <mesh position={points[11]}>
        <sphereGeometry args={[shoulderWidth * 0.102, 18, 14]} />
        <meshStandardMaterial color={topColor} roughness={style.roughness} metalness={style.metalness} />
      </mesh>
      {/* Right Deltoid cap */}
      <mesh position={points[12]}>
        <sphereGeometry args={[shoulderWidth * 0.102, 18, 14]} />
        <meshStandardMaterial color={topColor} roughness={style.roughness} metalness={style.metalness} />
      </mesh>

      {/* Crew-neck Collar Trim at base of neck */}
      <mesh
        position={new Vector3(torsoTop.x, torsoTop.y + shoulderWidth * 0.02, torsoTop.z)}
        quaternion={torsoRotation}
        scale={[shoulderWidth * 0.16, torsoLength * 0.045, shoulderWidth * 0.16]}
      >
        <cylinderGeometry args={[1, 1, 1, 20]} />
        <meshStandardMaterial color={style.topTrim} roughness={style.roughness} />
      </mesh>

      {/* 6. Pelvis Bar: Anchors hip joints */}
      <Bone
        start={points[23]}
        end={points[24]}
        radius={shoulderWidth * 0.075}
        color={bottomColor}
        roughness={style.roughness}
        metalness={style.metalness}
      />

      {/* 7. Upper Arms (Biceps) - Tapered from shoulder to elbow */}
      {/* Left upper arm (T-shirt short sleeve in casual) */}
      <Bone
        start={points[11]}
        end={points[13]}
        radiusTop={shoulderWidth * 0.086}
        radiusBottom={shoulderWidth * 0.070}
        color={upperArmColor}
        roughness={style.roughness}
        metalness={style.metalness}
      />
      {/* Right upper arm */}
      <Bone
        start={points[12]}
        end={points[14]}
        radiusTop={shoulderWidth * 0.086}
        radiusBottom={shoulderWidth * 0.070}
        color={upperArmColor}
        roughness={style.roughness}
        metalness={style.metalness}
      />

      {/* 8. Forearms - Tapered from elbow to wrist */}
      {/* Left forearm (Bare skin in casual, cyber conduit in future) */}
      <Bone
        start={points[13]}
        end={points[15]}
        radiusTop={shoulderWidth * 0.068}
        radiusBottom={shoulderWidth * 0.052}
        color={forearmColor}
        roughness={outfit === 'casual' ? 0.54 : style.roughness}
        metalness={outfit === 'casual' ? 0.05 : style.metalness}
      />
      {/* Right forearm */}
      <Bone
        start={points[14]}
        end={points[16]}
        radiusTop={shoulderWidth * 0.068}
        radiusBottom={shoulderWidth * 0.052}
        color={forearmColor}
        roughness={outfit === 'casual' ? 0.54 : style.roughness}
        metalness={outfit === 'casual' ? 0.05 : style.metalness}
      />

      {/* 9. Legs (Trousers/Tights) - Tapered thighs and calves */}
      {/* Left thigh */}
      <Bone
        start={points[23]}
        end={points[25]}
        radiusTop={shoulderWidth * 0.104}
        radiusBottom={shoulderWidth * 0.084}
        color={bottomColor}
        roughness={style.roughness}
        metalness={style.metalness}
      />
      {/* Left calf / shin */}
      <Bone
        start={points[25]}
        end={points[27]}
        radiusTop={shoulderWidth * 0.084}
        radiusBottom={shoulderWidth * 0.064}
        color={bottomColor}
        roughness={style.roughness}
        metalness={style.metalness}
      />

      {/* Right thigh */}
      <Bone
        start={points[24]}
        end={points[26]}
        radiusTop={shoulderWidth * 0.104}
        radiusBottom={shoulderWidth * 0.084}
        color={bottomColor}
        roughness={style.roughness}
        metalness={style.metalness}
      />
      {/* Right calf / shin */}
      <Bone
        start={points[26]}
        end={points[28]}
        radiusTop={shoulderWidth * 0.084}
        radiusBottom={shoulderWidth * 0.064}
        color={bottomColor}
        roughness={style.roughness}
        metalness={style.metalness}
      />

      {/* 10. Shoes / Sneakers */}
      <FootFigure
        ankle={points[27]}
        toe={points[31]}
        style={style}
        outfit={outfit}
        shoulderWidth={shoulderWidth}
      />
      <FootFigure
        ankle={points[28]}
        toe={points[32]}
        style={style}
        outfit={outfit}
        shoulderWidth={shoulderWidth}
      />

      {/* 11. Neck: Tapered cylinder connecting torsoTop to skull base */}
      <Bone
        start={new Vector3(torsoTop.x, torsoTop.y + shoulderWidth * 0.015, torsoTop.z)}
        end={new Vector3(headCenter.x, headCenter.y - headRadius * 0.82, headCenter.z)}
        radiusTop={shoulderWidth * 0.064}
        radiusBottom={shoulderWidth * 0.076}
        color={style.skin}
        roughness={0.52}
      />

      {/* 12. Head Figure */}
      <AvatarHead
        point={headCenter}
        tilt={headTilt}
        cues={faceCues}
        style={style}
        outfit={outfit}
        radius={headRadius}
      />

      {/* 13. Articulated Joint Knobs & Markers */}
      {/* Elbows */}
      {[13, 14].map((index) => (
        <Joint
          key={`elbow-${index}`}
          point={points[index]}
          radius={shoulderWidth * 0.070}
          color={outfit === 'casual' ? style.skin : style.accent}
          emissive={outfit === 'future' ? style.accent : undefined}
          emissiveIntensity={outfit === 'future' ? 0.6 : 0}
        />
      ))}
      {/* Wrists */}
      {[15, 16].map((index) => (
        <Joint
          key={`wrist-${index}`}
          point={points[index]}
          radius={shoulderWidth * 0.056}
          color={outfit === 'casual' ? style.skin : style.accent}
          emissive={outfit === 'future' ? style.accent : undefined}
          emissiveIntensity={outfit === 'future' ? 0.5 : 0}
        />
      ))}
      {/* Knees */}
      {[25, 26].map((index) => (
        <Joint
          key={`knee-${index}`}
          point={points[index]}
          radius={shoulderWidth * 0.082}
          color={outfit === 'casual' ? bottomColor : (outfit === 'future' ? style.accent : bottomColor)}
          emissive={outfit === 'future' ? style.accent : undefined}
          emissiveIntensity={outfit === 'future' ? 0.5 : 0}
        />
      ))}
      {/* Ankles */}
      {[27, 28].map((index) => (
        <Joint
          key={`ankle-${index}`}
          point={points[index]}
          radius={shoulderWidth * 0.062}
          color={outfit === 'casual' ? bottomColor : style.accent}
        />
      ))}

      {/* Mocap studio tracking marker spheres */}
      {active && outfit === 'mocap' && [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28].map((index) => (
        <Joint
          key={`mocap-marker-${index}`}
          point={points[index]}
          radius={shoulderWidth * 0.034}
          color={style.accent}
          emissive={style.accent}
          emissiveIntensity={0.65}
        />
      ))}

      {/* 14. Hands with 5-finger tracking */}
      {hands.map((hand, index) => (
        <HandFigure
          key={`${hand.handedness}-${index}`}
          hand={hand}
          avatarLandmarks={landmarks}
          calibration={calibration}
          scale={scale}
          color={handColor}
        />
      ))}
    </group>
  );
}

function Stage({
  landmarks,
  calibration,
  hands,
  faceCues,
  outfit,
  active,
  transparent = false,
}: DemoAvatarProps) {
  return (
    <>
      {!transparent && <color attach="background" args={['#101b21']} />}
      <ambientLight intensity={1.4} />
      <directionalLight position={[3, 5, 5]} intensity={2.2} color="#ffffff" />
      <directionalLight position={[-3, 2, 3]} intensity={1.2} color="#c8e4ff" />
      <pointLight position={[0, -2, 3]} intensity={5.5} color="#ffe8cc" />
      <AvatarFigure
        landmarks={landmarks}
        calibration={calibration}
        hands={hands}
        faceCues={faceCues}
        outfit={outfit}
        active={active}
      />
    </>
  );
}

let webGLSupportCache: boolean | undefined;

function supportsWebGL() {
  if (webGLSupportCache !== undefined) return webGLSupportCache;
  try {
    const canvas = document.createElement('canvas');
    const context =
      canvas.getContext('webgl2') ?? canvas.getContext('webgl');
    webGLSupportCache = Boolean(context);
    context?.getExtension('WEBGL_lose_context')?.loseContext();
  } catch {
    webGLSupportCache = false;
  }
  return webGLSupportCache;
}

const SVG_HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15],
  [15, 16], [0, 17], [17, 18], [18, 19], [19, 20],
];

function SvgAvatarFallback({
  landmarks,
  calibration,
  hands,
  faceCues,
  outfit,
  active,
}: DemoAvatarProps) {
  const style = OUTFIT_STYLES[outfit];
  const scale = getAvatarCalibrationScale(calibration);
  const point = (index: number) => {
    const source = getValidLandmark(
      index,
      landmarks,
      NEUTRAL[index] ?? { x: 0, y: 0, z: 0 },
    );
    const mapped = mapLandmarkToAvatar(source, calibration, scale);
    return { x: 100 + mapped.x * 90, y: 160 - mapped.y * 90 };
  };
  const handPoint = (hand: TrackedHand, index: number) => {
    const worldPoint = hand.worldLandmarks?.[index];
    const worldOrigin = hand.worldLandmarks?.[0];
    const wristIndex = hand.handedness === 'Left' ? 15 : 16;
    const poseWrist = landmarks[wristIndex] ?? NEUTRAL[wristIndex];
    const raw = worldPoint && worldOrigin
      ? {
          ...worldPoint,
          x: poseWrist.x + worldPoint.x - worldOrigin.x,
          y: poseWrist.y + worldPoint.y - worldOrigin.y,
          z: poseWrist.z - (worldPoint.z - worldOrigin.z),
        }
      : legacyImagePointToWorld(hand.landmarks[index]);
    const mapped = mapLandmarkToAvatar(
      raw,
      worldPoint ? calibration : null,
      worldPoint ? scale : undefined,
    );
    return { x: 100 + mapped.x * 90, y: 160 - mapped.y * 90 };
  };
  const midpoint = (left: number, right: number) => {
    const a = point(left);
    const b = point(right);
    return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  };
  const head = point(0);
  const shoulders = midpoint(11, 12);
  const hips = midpoint(23, 24);
  const earLeft = point(7);
  const earRight = point(8);
  const tilt =
    (Math.atan2(earRight.y - earLeft.y, earRight.x - earLeft.x) * 180) /
    Math.PI;
  const expression = faceCues;
  const mouthHeight = Math.max(1, 1 + (expression?.jawOpen ?? 0) * 6);

  const topColor = style.top;
  const bottomColor = style.bottom;
  const forearmColor = outfit === 'casual' ? style.skin : style.top;
  const handColor = outfit === 'casual' ? style.skin : style.accent;

  const segments = [
    { start: point(11), end: point(12), width: 24, color: topColor },
    { start: point(23), end: point(24), width: 22, color: bottomColor },
    { start: shoulders, end: hips, width: 28, color: topColor },
    { start: point(11), end: point(13), width: 14, color: topColor },
    { start: point(13), end: point(15), width: 11, color: forearmColor },
    { start: point(12), end: point(14), width: 14, color: topColor },
    { start: point(14), end: point(16), width: 11, color: forearmColor },
    { start: point(23), end: point(25), width: 16, color: bottomColor },
    { start: point(25), end: point(27), width: 13, color: bottomColor },
    { start: point(24), end: point(26), width: 16, color: bottomColor },
    { start: point(26), end: point(28), width: 13, color: bottomColor },
    { start: point(27), end: point(31), width: 11, color: style.shoes },
    { start: point(28), end: point(32), width: 11, color: style.shoes },
  ];

  const drawLine = (start: { x: number; y: number }, end: { x: number; y: number }, width: number, color: string, key: string) => (
    <line
      key={key}
      x1={start.x}
      y1={start.y}
      x2={end.x}
      y2={end.y}
      stroke={color}
      strokeWidth={width}
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  );

  return (
    <svg viewBox="0 0 200 320" preserveAspectRatio="xMidYMid meet" role="img" aria-label={`Configurable demo avatar wearing ${outfit}`}>
      <g>
        {segments.map((segment, index) =>
          drawLine(segment.start, segment.end, segment.width, segment.color, `segment-${index}`),
        )}
        <line
          x1={shoulders.x}
          y1={shoulders.y}
          x2={head.x}
          y2={head.y + 12}
          stroke={style.skin}
          strokeWidth="11"
          strokeLinecap="round"
        />
        {hands.map((hand, handIndex) =>
          SVG_HAND_CONNECTIONS.map(([from, to]) => {
            return drawLine(
              handPoint(hand, from),
              handPoint(hand, to),
              3.2,
              handColor,
              `hand-${handIndex}-${from}-${to}`,
            );
          }),
        )}
        <g transform={`rotate(${tilt} ${head.x} ${head.y})`}>
          {/* Hair back */}
          <circle cx={head.x} cy={head.y - 2} r="18.5" fill={style.hair} />
          {/* Cranium */}
          <circle cx={head.x} cy={head.y} r="17" fill={style.skin} />
          {/* Hair fringe */}
          <path d={`M${head.x - 16} ${head.y - 4} Q${head.x - 12} ${head.y - 22} ${head.x + 5} ${head.y - 17} Q${head.x + 17} ${head.y - 15} ${head.x + 17} ${head.y - 2} Q${head.x + 7} ${head.y - 12} ${head.x - 16} ${head.y - 4}`} fill={style.hair} />
          {/* Ears */}
          <circle cx={head.x - 17} cy={head.y + 1} r="3.2" fill={style.skin} />
          <circle cx={head.x + 17} cy={head.y + 1} r="3.2" fill={style.skin} />
          {/* Eyebrows */}
          <line x1={head.x - 9} y1={head.y - 3} x2={head.x - 3} y2={head.y - 3} stroke={style.hair} strokeWidth="1.8" strokeLinecap="round" />
          <line x1={head.x + 3} y1={head.y - 3} x2={head.x + 9} y2={head.y - 3} stroke={style.hair} strokeWidth="1.8" strokeLinecap="round" />
          {/* Eyes */}
          <ellipse cx={head.x - 6} cy={head.y + 1} rx="2.5" ry={Math.max(0.6, 2.5 * (1 - (expression?.leftBlink ?? 0) * 0.88))} fill="#fcfaf4" />
          <circle cx={head.x - 6} cy={head.y + 1} r="1.3" fill="#3a6888" />
          <ellipse cx={head.x + 6} cy={head.y + 1} rx="2.5" ry={Math.max(0.6, 2.5 * (1 - (expression?.rightBlink ?? 0) * 0.88))} fill="#fcfaf4" />
          <circle cx={head.x + 6} cy={head.y + 1} r="1.3" fill="#3a6888" />
          {/* Nose */}
          <ellipse cx={head.x} cy={head.y + 4.5} rx="1.6" ry="1.2" fill={style.skin} stroke="#c08465" strokeWidth="0.8" />
          {/* Mouth */}
          {(expression?.surprise ?? 0) > 0.25 ? (
            <ellipse cx={head.x} cy={head.y + 10} rx={3} ry={mouthHeight} fill="#381518" stroke="#754a48" strokeWidth="1.2" />
          ) : (expression?.frown ?? 0) > 0.25 ? (
            <path
              d={`M${head.x - 5} ${head.y + 11 + (expression?.frown ?? 0) * 2} Q${head.x} ${head.y + 8 - (expression?.frown ?? 0) * 1.5} ${head.x + 5} ${head.y + 11 + (expression?.frown ?? 0) * 2}`}
              stroke="#754a48"
              strokeWidth="2.2"
              fill="none"
              strokeLinecap="round"
            />
          ) : (
            <path
              d={`M${head.x - 5 - (expression?.smile ?? 0) * 3} ${head.y + 9 - (expression?.smile ?? 0) * 2.5} Q${head.x} ${head.y + 9 + (expression?.smile ?? 0) * 3.5} ${head.x + 5 + (expression?.smile ?? 0) * 3} ${head.y + 9 - (expression?.smile ?? 0) * 2.5}`}
              stroke="#754a48"
              strokeWidth="2.2"
              fill={((expression?.smile ?? 0) > 0.35) ? '#f8fafc' : 'none'}
              strokeLinecap="round"
            />
          )}
        </g>
        {active && outfit === 'mocap' && [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28].map((index) => {
          const joint = point(index);
          return <circle key={`marker-${index}`} cx={joint.x} cy={joint.y} r="3.5" fill="#3ef09e" stroke="#1e252b" strokeWidth="1.5" />;
        })}
      </g>
    </svg>
  );
}

export function DemoAvatar({
  landmarks,
  calibration,
  hands,
  faceCues,
  outfit,
  active,
  transparent = false,
}: DemoAvatarProps) {
  const [webGLAvailable] = useState(supportsWebGL);
  if (!webGLAvailable) {
    return (
      <SvgAvatarFallback
        landmarks={landmarks}
        calibration={calibration}
        hands={hands}
        faceCues={faceCues}
        outfit={outfit}
        active={active}
        transparent={transparent}
      />
    );
  }
  return (
    <Canvas
      camera={{ position: [0, -0.05, 4.8], fov: 40 }}
      dpr={[1, 1.75]}
      gl={{ antialias: true, alpha: transparent, powerPreference: 'high-performance' }}
      onCreated={({ scene, camera }) => {
        scene.background = transparent ? null : new Color('#101b21');
        camera.lookAt(0, -0.05, 0);
      }}
    >
      <Stage
        landmarks={landmarks}
        calibration={calibration}
        hands={hands}
        faceCues={faceCues}
        outfit={outfit}
        active={active}
        transparent={transparent}
      />
    </Canvas>
  );
}
