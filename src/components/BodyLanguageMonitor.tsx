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

const MIN_GESTURE_INTERVAL_MS = 900;
const MIN_NOD_INTERVAL_MS = 700;

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

const buildSuggestions = (postureScore: number, handGestureRate: number, headNodCount: number) => {
  const suggestions: string[] = [];

  if (postureScore < 70) {
    suggestions.push('Try to keep your shoulders level and stay more upright.');
  } else {
    suggestions.push('Your posture looks steady. Keep that balanced stance.');
  }

  if (handGestureRate < 6) {
    suggestions.push('Use a few more deliberate hand gestures to support key points.');
  } else if (handGestureRate > 30) {
    suggestions.push('Your hands are active. Slow the gestures down so they feel intentional.');
  } else {
    suggestions.push('Your hand gestures are in a healthy range for presentation practice.');
  }

  if (headNodCount > 18) {
    suggestions.push('You are nodding often. Pause the head movement so the delivery feels calmer.');
  } else {
    suggestions.push('Head movement looks controlled and should not distract the audience.');
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
  const lastNoseYRef = useRef<number | null>(null);
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
      if (video) {
        video.srcObject = null;
      }

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
        lastNoseYRef.current = null;

        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas) {
          throw new Error('Camera elements are not ready.');
        }

        video.srcObject = stream;
        await video.play();

        const holistic = new Holistic({
          locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
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
          if (!canvasRef.current || !videoRef.current) {
            return;
          }

          const ctx = canvasRef.current.getContext('2d');
          if (!ctx) {
            return;
          }

          const width = videoRef.current.videoWidth || 640;
          const height = videoRef.current.videoHeight || 360;
          canvasRef.current.width = width;
          canvasRef.current.height = height;

          ctx.save();
          ctx.clearRect(0, 0, width, height);
          ctx.drawImage(results.image, 0, 0, width, height);

          const now = Date.now();

          if (results.poseLandmarks) {
            const leftShoulder = results.poseLandmarks[11];
            const rightShoulder = results.poseLandmarks[12];
            const nose = results.poseLandmarks[0];

            if (leftShoulder && rightShoulder) {
              const posture = clamp(100 - (Math.abs(leftShoulder.y - rightShoulder.y) / 0.12) * 100, 0, 100);
              postureSamplesRef.current += 1;
              postureTotalRef.current += posture;

              ctx.strokeStyle = posture > 70 ? '#10b981' : '#f59e0b';
              ctx.lineWidth = 3;
              ctx.beginPath();
              ctx.moveTo(leftShoulder.x * width, leftShoulder.y * height);
              ctx.lineTo(rightShoulder.x * width, rightShoulder.y * height);
              ctx.stroke();
            }

            if (nose) {
              if (
                lastNoseYRef.current !== null &&
                lastNoseYRef.current - nose.y > 0.03 &&
                now - lastNodAtRef.current > MIN_NOD_INTERVAL_MS
              ) {
                nodCountRef.current += 1;
                lastNodAtRef.current = now;
              }
              lastNoseYRef.current = nose.y;
            }
          }

          if ((results.leftHandLandmarks || results.rightHandLandmarks) && now - lastGestureAtRef.current > MIN_GESTURE_INTERVAL_MS) {
            gestureCountRef.current += 1;
            lastGestureAtRef.current = now;
          }

          ctx.fillStyle = 'rgba(15, 23, 42, 0.72)';
          ctx.fillRect(12, 12, 180, 72);
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
        <video
          ref={videoRef}
          className="hidden"
          autoPlay
          muted
          playsInline
        />
        <canvas
          ref={canvasRef}
          className="block h-48 w-full object-cover"
        />
      </div>
      <p className="text-xs text-gray-500">{status}</p>
    </div>
  );
};

export default BodyLanguageMonitor;
