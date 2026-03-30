import { useEffect, useRef, useState } from 'react';
import { Camera } from '@mediapipe/camera_utils';
import { Holistic } from '@mediapipe/holistic';

export interface BodyMetrics {
  postureScore: number;
  handGestureRate: number;
  headNodCount: number;
  suggestions: string[];
}

interface BodyLanguageMonitorProps {
  active: boolean;
  onMetricsChange: (metrics: BodyMetrics | null) => void;
}

const MIN_GESTURE_INTERVAL_MS = 1000;
const MIN_NOD_INTERVAL_MS = 900;
const LANDMARK_SMOOTHING = 0.65;

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

const distance2D = (a: any, b: any) => {
  if (!a || !b) return 0;
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
};

const midpoint = (a: any, b: any) => ({
  x: (a.x + b.x) / 2,
  y: (a.y + b.y) / 2,
});

const smoothValue = (prev: number | null, next: number, alpha = LANDMARK_SMOOTHING) => {
  if (prev === null) return next;
  return prev * alpha + next * (1 - alpha);
};

const buildSuggestions = (
  postureScore: number,
  handGestureRate: number,
  headNodCount: number
) => {
  const suggestions: string[] = [];

  if (postureScore < 60) {
    suggestions.push('Sit or stand taller, keep your shoulders level, and center your head.');
  } else if (postureScore < 80) {
    suggestions.push('Your posture is decent, but try to stay a little more upright and balanced.');
  } else {
    suggestions.push('Your posture looks strong and steady.');
  }

  if (handGestureRate < 4) {
    suggestions.push('Use a few more intentional hand gestures to emphasize key ideas.');
  } else if (handGestureRate > 16) {
    suggestions.push('Your gestures are a bit frequent. Slow them down to feel more deliberate.');
  } else {
    suggestions.push('Your gesture rate looks natural and supportive.');
  }

  if (headNodCount > 10) {
    suggestions.push('You are nodding quite often. Reduce repeated head movement for a calmer delivery.');
  } else {
    suggestions.push('Your head movement looks controlled.');
  }

  return suggestions;
};

const BodyLanguageMonitor = ({ active, onMetricsChange }: BodyLanguageMonitorProps) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const cameraRef = useRef<any>(null);
  const holisticRef = useRef<any>(null);

  const startedAtRef = useRef<number>(0);
  const gestureCountRef = useRef<number>(0);
  const nodCountRef = useRef<number>(0);
  const postureSamplesRef = useRef<number>(0);
  const postureTotalRef = useRef<number>(0);

  const lastGestureAtRef = useRef<number>(0);
  const lastNodAtRef = useRef<number>(0);

  const smoothedNoseYRef = useRef<number | null>(null);
  const prevSmoothedNoseYRef = useRef<number | null>(null);

  const leftHandAnchorRef = useRef<any>(null);
  const rightHandAnchorRef = useRef<any>(null);

  const nodStateRef = useRef<'neutral' | 'down'>('neutral');

  const [status, setStatus] = useState('Camera idle');

  useEffect(() => {
    let cancelled = false;

    const stopMonitoring = async () => {
      cameraRef.current?.stop();
      cameraRef.current = null;

      if (holisticRef.current) {
        await holisticRef.current.close();
        holisticRef.current = null;
      }

      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }

      const video = videoRef.current;
      if (video) video.srcObject = null;

      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx?.clearRect(0, 0, canvas.width, canvas.height);
      }
    };

    if (!active) {
      setStatus('Camera idle');
      onMetricsChange(null);
      void stopMonitoring();
      return;
    }

    const startMonitoring = async () => {
      try {
        setStatus('Requesting camera access...');

        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 360, facingMode: 'user' },
          audio: false,
        });

        if (cancelled) {
          stream.getTracks().forEach(track => track.stop());
          return;
        }

        streamRef.current = stream;
        startedAtRef.current = Date.now();
        gestureCountRef.current = 0;
        nodCountRef.current = 0;
        postureSamplesRef.current = 0;
        postureTotalRef.current = 0;
        lastGestureAtRef.current = 0;
        lastNodAtRef.current = 0;

        smoothedNoseYRef.current = null;
        prevSmoothedNoseYRef.current = null;
        leftHandAnchorRef.current = null;
        rightHandAnchorRef.current = null;
        nodStateRef.current = 'neutral';

        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas) {
          throw new Error('Camera elements are not ready.');
        }

        video.srcObject = stream;
        await video.play();

        const holistic = new Holistic({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
        });
        holisticRef.current = holistic;

        holistic.setOptions({
          modelComplexity: 1,
          smoothLandmarks: true,
          refineFaceLandmarks: true,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        holistic.onResults((results: any) => {
          if (!canvasRef.current || !videoRef.current) return;

          const ctx = canvasRef.current.getContext('2d');
          if (!ctx) return;

          const width = videoRef.current.videoWidth || 640;
          const height = videoRef.current.videoHeight || 360;
          canvasRef.current.width = width;
          canvasRef.current.height = height;

          ctx.save();
          ctx.clearRect(0, 0, width, height);
          ctx.drawImage(results.image, 0, 0, width, height);

          const now = Date.now();

          // ------------------------
          // SMARTER POSTURE SCORING
          // ------------------------
          if (results.poseLandmarks) {
            const pose = results.poseLandmarks;
            const nose = pose[0];
            const leftShoulder = pose[11];
            const rightShoulder = pose[12];
            const leftHip = pose[23];
            const rightHip = pose[24];

            if (leftShoulder && rightShoulder && leftHip && rightHip && nose) {
              const shoulderMid = midpoint(leftShoulder, rightShoulder);
              const hipMid = midpoint(leftHip, rightHip);

              const shoulderTilt = Math.abs(leftShoulder.y - rightShoulder.y); // smaller is better
              const torsoOffsetX = Math.abs(shoulderMid.x - hipMid.x); // smaller is better
              const headOffsetX = Math.abs(nose.x - shoulderMid.x); // centered head is better

              const shoulderScore = clamp(100 - (shoulderTilt / 0.10) * 100, 0, 100);
              const torsoScore = clamp(100 - (torsoOffsetX / 0.12) * 100, 0, 100);
              const headScore = clamp(100 - (headOffsetX / 0.10) * 100, 0, 100);

              const posture =
                Math.round(shoulderScore * 0.45 + torsoScore * 0.35 + headScore * 0.20);

              postureSamplesRef.current += 1;
              postureTotalRef.current += posture;

              ctx.strokeStyle = posture > 75 ? '#10b981' : posture > 55 ? '#f59e0b' : '#ef4444';
              ctx.lineWidth = 3;
              ctx.beginPath();
              ctx.moveTo(leftShoulder.x * width, leftShoulder.y * height);
              ctx.lineTo(rightShoulder.x * width, rightShoulder.y * height);
              ctx.stroke();

              ctx.beginPath();
              ctx.moveTo(shoulderMid.x * width, shoulderMid.y * height);
              ctx.lineTo(hipMid.x * width, hipMid.y * height);
              ctx.stroke();
            }

            // ------------------------
            // SMARTER NOD DETECTION
            // ------------------------
            if (nose) {
              const smoothed = smoothValue(smoothedNoseYRef.current, nose.y);
              prevSmoothedNoseYRef.current = smoothedNoseYRef.current;
              smoothedNoseYRef.current = smoothed;

              if (prevSmoothedNoseYRef.current !== null) {
                const deltaY = smoothed - prevSmoothedNoseYRef.current;

                // down motion
                if (deltaY > 0.008 && nodStateRef.current === 'neutral') {
                  nodStateRef.current = 'down';
                }

                // up return completes nod
                if (
                  deltaY < -0.008 &&
                  nodStateRef.current === 'down' &&
                  now - lastNodAtRef.current > MIN_NOD_INTERVAL_MS
                ) {
                  nodCountRef.current += 1;
                  lastNodAtRef.current = now;
                  nodStateRef.current = 'neutral';
                }
              }
            }
          }

          // ------------------------
          // SMARTER GESTURE DETECTION
          // ------------------------
          const detectHandGesture = (handLandmarks: any, anchorRef: React.MutableRefObject<any>) => {
            if (!handLandmarks || handLandmarks.length === 0) return false;

            const wrist = handLandmarks[0];
            const indexTip = handLandmarks[8];
            const pinkyTip = handLandmarks[20];

            if (!wrist || !indexTip || !pinkyTip) return false;

            const handCenter = {
              x: (wrist.x + indexTip.x + pinkyTip.x) / 3,
              y: (wrist.y + indexTip.y + pinkyTip.y) / 3,
            };

            if (!anchorRef.current) {
              anchorRef.current = handCenter;
              return false;
            }

            const motion = distance2D(handCenter, anchorRef.current);
            anchorRef.current = {
              x: anchorRef.current.x * 0.7 + handCenter.x * 0.3,
              y: anchorRef.current.y * 0.7 + handCenter.y * 0.3,
            };

            return motion > 0.035;
          };

          const leftGesture = detectHandGesture(results.leftHandLandmarks, leftHandAnchorRef);
          const rightGesture = detectHandGesture(results.rightHandLandmarks, rightHandAnchorRef);

          if ((leftGesture || rightGesture) && now - lastGestureAtRef.current > MIN_GESTURE_INTERVAL_MS) {
            gestureCountRef.current += 1;
            lastGestureAtRef.current = now;
          }

          // ------------------------
          // METRICS + UI
          // ------------------------
          ctx.fillStyle = 'rgba(15, 23, 42, 0.72)';
          ctx.fillRect(12, 12, 205, 76);
          ctx.fillStyle = '#ffffff';
          ctx.font = '14px sans-serif';

          const elapsedMinutes = Math.max((now - startedAtRef.current) / 60000, 1 / 60);
          const postureScore = postureSamplesRef.current
            ? Math.round(postureTotalRef.current / postureSamplesRef.current)
            : 0;
          const handGestureRate = Math.round(gestureCountRef.current / elapsedMinutes);
          const headNodCount = Math.round(nodCountRef.current / elapsedMinutes);

          const metrics: BodyMetrics = {
            postureScore,
            handGestureRate,
            headNodCount,
            suggestions: buildSuggestions(postureScore, handGestureRate, headNodCount),
          };

          ctx.fillText(`Posture: ${metrics.postureScore}%`, 24, 36);
          ctx.fillText(`Gestures/min: ${metrics.handGestureRate}`, 24, 56);
          ctx.fillText(`Nods/min: ${metrics.headNodCount}`, 24, 76);
          ctx.restore();

          onMetricsChange(metrics);
        });

        const camera = new Camera(video, {
          onFrame: async () => {
            if (holisticRef.current && videoRef.current) {
              await holisticRef.current.send({ image: videoRef.current });
            }
          },
          width: 640,
          height: 360,
        });

        cameraRef.current = camera;
        await camera.start();
        setStatus('Camera active');
      } catch (error) {
        console.error('Failed to start body language monitoring', error);
        setStatus('Camera unavailable. Check browser permissions and close other apps using the webcam.');
        onMetricsChange(null);
        await stopMonitoring();
      }
    };

    void startMonitoring();

    return () => {
      cancelled = true;
      void stopMonitoring();
    };
  }, [active, onMetricsChange]);

  return (
    <div className="flex flex-col gap-2">
      <div className="relative overflow-hidden rounded-md bg-slate-950">
        <video ref={videoRef} className="hidden" autoPlay muted playsInline />
        <canvas ref={canvasRef} className="block h-48 w-full object-cover" />
      </div>
      <p className="text-xs text-gray-500">{status}</p>
    </div>
  );
};

export default BodyLanguageMonitor;