# MotionMirror

MotionMirror is a browser-based motion capture studio. It uses the webcam only after the user starts tracking, processes camera frames in the browser, mirrors pose/hand/face-expression data onto a configurable demo avatar, and saves motion sessions locally.

## Current features

- Camera permission, live preview, start/stop, clear camera errors, and tracking-loss status.
- MediaPipe Pose Landmarker body points, confidence, adjustable smoothing response, and neutral-pose calibration.
- MediaPipe Hand Landmarker for up to two hands and finger landmarks.
- MediaPipe Face Landmarker expression blendshape cues for smile, frown, surprise, neutral-looking expression, jaw opening, and blinking.
- A procedural demo avatar that follows tracked head/body/arm/hand points and supported facial cues.
- Three outfit presets: motion-capture, casual, and futuristic.
- Recording start, pause, resume, stop/save, local persistence, playback controls, timeline seeking, restart, and delete confirmation.
- JSON export of a saved recording.
- An SVG avatar fallback for browsers or preview environments without WebGL.

## Run on Windows

Install Node.js 24 and enable Corepack, then open PowerShell at the workspace root:

```powershell
corepack enable
pnpm install
$env:PORT = "5173"
$env:BASE_PATH = "/"
pnpm --filter @workspace/motionmirror run dev
```

Open the local Vite URL printed in the terminal. Camera access is allowed on `localhost`; on a hosted deployment, use HTTPS. The browser asks for camera permission only after **Start camera** is pressed.

Check the app:

```powershell
pnpm --filter @workspace/motionmirror run typecheck
$env:PORT = "5173"
$env:BASE_PATH = "/"
pnpm --filter @workspace/motionmirror run build
```

The Replit preview uses the managed `artifacts/motionmirror: web` workflow, which supplies its own port and base path.

## Main files

```text
src/
  App.tsx                         Studio interface and session controls
  avatar/DemoAvatar.tsx           Procedural avatar and SVG fallback
  tracking/types.ts               Tracking and calibration types
  tracking/usePoseTracking.ts     Browser camera and MediaPipe pipeline
  recording/types.ts              Versioned motion-session format
  recording/storage.ts            IndexedDB storage and validation
  recording/useMotionRecorder.ts  Capture, playback, and timeline state
  index.css                       Responsive interface styling
```

## Motion recording format

Recordings use format version `1` and are stored in IndexedDB under the `motionmirror-local` database. The app stores motion data only; it does not store camera video.

```json
{
  "version": 1,
  "id": "browser-generated-id",
  "name": "Take · date and time",
  "createdAt": "ISO-8601 timestamp",
  "durationMs": 4600,
  "outfit": "mocap",
  "sampleRate": 10,
  "frames": [
    {
      "timeMs": 0,
      "landmarks": [{ "x": 0.5, "y": 0.3, "z": 0, "visibility": 0.95 }],
      "hands": [
        {
          "handedness": "Left",
          "landmarks": [{ "x": 0.5, "y": 0.3, "z": 0 }]
        }
      ],
      "faceCues": {
        "smile": 0.1,
        "frown": 0.02,
        "surprise": 0.05,
        "neutral": 0.9,
        "jawOpen": 0.02,
        "leftBlink": 0,
        "rightBlink": 0,
        "landmarkCount": 478
      },
      "confidence": 0.9
    }
  ]
}
```

Landmark coordinates are normalized to the camera image; `z` is the model's relative depth. Playback interpolates recorded samples to preserve timing. The selected outfit is stored with each take. Export downloads the same versioned JSON structure.

## Processing and network behavior

There is no server-side tracking engine or WebSocket protocol: the webcam stream, MediaPipe inference, avatar rendering, and recording/playback state run directly in the browser. Camera frames and saved motion sessions are not transmitted to MotionMirror servers. The MediaPipe WASM runtime and model files are fetched from public CDNs when tracking is started, so an internet connection is required for initial model loading.

The app attempts GPU inference and falls back to CPU when GPU initialization fails. Hand and face tracking are optional: if either task cannot load, body pose tracking continues and the UI reports the missing feature.

## Dependencies

Key direct dependencies:

- `@mediapipe/tasks-vision` `^1.1.1`
- `three` `^0.186.1`
- `@react-three/fiber` `^9.8.2`
- `@types/three` `^0.186.0`

The workspace lockfile pins the resolved transitive dependency versions. No paid API, API key, or external account is required.

## Limitations

- Webcam input does not automatically reconstruct a user's identity. The current avatar is a clearly labeled procedural demo figure, not a personalized likeness.
- Outfit presets adjust procedural materials and tracking markers. Separate production-quality rigged wardrobe meshes are not included.
- Body and hand landmarks are retargeted to the demo figure; it is not a production humanoid rig with a full joint hierarchy.
- Face animation is a best-effort mapping from available blendshape cues to simple avatar controls. The percentages indicate cue activation, not confidence about a person's internal emotional state.
- Tracking quality depends on lighting, camera placement, occlusion, and device performance. The app does not claim a camera-and-device hardware test unless one is performed explicitly.

## Verification

- App TypeScript check: run `pnpm --filter @workspace/motionmirror run typecheck`.
- Local production bundle: run the build command above.
- Browser preview rendering: checked through the Replit preview.
- Actual webcam permission, tracking quality, and device performance must be verified in a browser with a camera; the preview capture environment does not grant webcam access.
