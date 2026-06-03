import { AlertLevel, DetectionStats, ProctorAlert } from '@/types';

const normalizeBaseUrl = (url: string) => url.replace(/\/+$/, '');

const DEFAULT_AI_SERVER_URL = 'https://proctorclay-ai-server.onrender.com';

const AI_SERVER_URL = (() => {
  const configured = import.meta.env.VITE_AI_SERVER_URL?.trim();
  if (configured) return normalizeBaseUrl(configured);
  if (import.meta.env.DEV) return 'http://localhost:5000';
  return DEFAULT_AI_SERVER_URL;
})();

export interface CameraDevice {
  index: number;
  name: string;
  resolution: string;
  deviceId: string;
}

export interface HealthDetails {
  status: string;
  model_loaded: boolean;
  mediapipe_loaded: boolean;
  stream_active: boolean;
  webcam_active: boolean;
  current_stream: string | null;
}

export interface DetectionItem {
  class: string;
  exam_type?: string;
  confidence: number;
  raw_confidence?: number;
  bbox: [number, number, number, number];
}

export interface AIDetection {
  frame_number: number;
  timestamp: number;
  detections: DetectionItem[];
  person_count: number;
  prohibited_items: DetectionItem[];
  behaviors: Array<{
    type: string;
    confidence: number;
    yaw?: number;
    pitch?: number;
    distance?: number;
  }>;
  head_poses?: Array<{
    yaw: number;
    pitch: number;
    roll: number;
    looking_away: boolean;
    looking_down: boolean;
    head_turn: boolean;
    bbox: [number, number, number, number];
  }>;
}

export interface FrameResponse {
  frame: string;
  result: AIDetection | null;
  timestamp: number;
}

export interface DetectedFrameResponse extends AIDetection {
  annotated_frame?: string;
}

const DETECTION_POINTS: Record<string, number> = {
  PHONE: 30,
  CHIT: 25,
  TEXTBOOK: 30,
  NOTEBOOK: 20,
  DEVICE: 15,
  HEAD_TURN: 10,
  LEANING: 10,
  MULTIPLE_PEOPLE: 25,
  NO_PERSON: 40,
};

const clamp = (value: number) => Math.max(0, Math.min(100, value));

const requestJson = async <T>(path: string, init?: RequestInit): Promise<T> => {
  if (!AI_SERVER_URL) {
    throw new Error('VITE_AI_SERVER_URL is not configured');
  }

  const response = await fetch(`${AI_SERVER_URL}${path}`, {
    ...init,
    headers: {
      ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
      ...(init?.headers || {}),
    },
  });

  if (!response.ok) {
    const message = await response.text().catch(() => '');
    throw new Error(message || `Request failed with status ${response.status}`);
  }

  return response.json() as Promise<T>;
};

const mapBehaviorType = (type: string): keyof typeof DETECTION_POINTS | null => {
  switch (type) {
    case 'HEAD_TURN':
      return 'HEAD_TURN';
    case 'LOOKING_DOWN':
      return 'LEANING';
    case 'LOOKING_AWAY':
      return 'HEAD_TURN';
    case 'PROXIMITY_ALERT':
      return 'MULTIPLE_PEOPLE';
    default:
      return null;
  }
};

const mapDetectionType = (item: DetectionItem): keyof typeof DETECTION_POINTS | null => {
  if (item.exam_type && item.exam_type in DETECTION_POINTS) {
    return item.exam_type as keyof typeof DETECTION_POINTS;
  }

  switch (item.class) {
    case 'cell phone':
      return 'PHONE';
    case 'book':
      return 'TEXTBOOK';
    case 'laptop':
    case 'keyboard':
    case 'mouse':
    case 'remote':
    case 'tv':
      return 'DEVICE';
    default:
      return null;
  }
};

const toAlertLevel = (score: number): AlertLevel => {
  if (score >= 40) return AlertLevel.CRITICAL;
  if (score >= 30) return AlertLevel.HIGH;
  if (score >= 20) return AlertLevel.MEDIUM;
  return AlertLevel.LOW;
};

const getSeatLabel = (bbox?: [number, number, number, number]) => {
  if (!bbox) return 'S1';

  const centerX = bbox[0] + bbox[2] / 2;
  const seatNumber = Math.max(1, Math.round(centerX / 120));
  return `S${seatNumber}`;
};

export const aiService = {
  getBaseUrl: () => AI_SERVER_URL,

  getHealthDetails: async (): Promise<HealthDetails | null> => {
    try {
      return await requestJson<HealthDetails>('/health');
    } catch {
      return null;
    }
  },

  listCameras: async (): Promise<CameraDevice[]> => {
    try {
      if (!navigator.mediaDevices?.enumerateDevices) {
        return [];
      }

      const devices = await navigator.mediaDevices.enumerateDevices();
      return devices
        .filter(device => device.kind === 'videoinput')
        .map((device, index) => ({
          index,
          name: device.label || `Camera ${index + 1}`,
          resolution: 'Browser camera',
          deviceId: device.deviceId,
        }));
    } catch {
      return [];
    }
  },

  startWebcam: async (cameraIndex = 0): Promise<{ status: string }> =>
    requestJson<{ status: string }>('/webcam/start', {
      method: 'POST',
      body: JSON.stringify({ camera_index: cameraIndex }),
    }),

  stopWebcam: async (): Promise<{ status: string }> =>
    requestJson<{ status: string }>('/webcam/stop', { method: 'POST' }),

  getWebcamFrame: async (): Promise<FrameResponse | null> => {
    try {
      return await requestJson<FrameResponse>('/webcam/frame');
    } catch {
      return null;
    }
  },

  startStream: async (streamUrl: string): Promise<{ status: string }> =>
    requestJson<{ status: string }>('/stream/start', {
      method: 'POST',
      body: JSON.stringify({ stream_url: streamUrl }),
    }),

  stopStream: async (): Promise<{ status: string }> =>
    requestJson<{ status: string }>('/stream/stop', { method: 'POST' }),

  getFrame: async (): Promise<FrameResponse | null> => {
    try {
      return await requestJson<FrameResponse>('/stream/frame');
    } catch {
      return null;
    }
  },

  analyzeFrame: async (base64Image: string): Promise<DetectedFrameResponse> =>
    requestJson<DetectedFrameResponse>('/detect/frame', {
      method: 'POST',
      body: JSON.stringify({ image: base64Image }),
    }),

  calculateIntegrityScore: (result: AIDetection): number => {
    let score = 100;

    result.prohibited_items.forEach(item => {
      const type = mapDetectionType(item);
      if (type) {
        score -= DETECTION_POINTS[type];
      }
    });

    result.behaviors.forEach(behavior => {
      const type = mapBehaviorType(behavior.type);
      if (type) {
        score -= DETECTION_POINTS[type];
      }
    });

    if (result.person_count === 0) {
      score -= DETECTION_POINTS.NO_PERSON;
    }

    return clamp(score);
  },

  getDetectionStats: (result: AIDetection): DetectionStats => {
    const stats: DetectionStats = {
      phone: 0,
      chit: 0,
      textbook: 0,
      notebook: 0,
      device: 0,
      headTurn: 0,
      leaning: 0,
      multiplePeople: 0,
      detectedCount: result.person_count,
      expectedCount: 0,
    };

    result.prohibited_items.forEach(item => {
      const type = mapDetectionType(item);
      switch (type) {
        case 'PHONE':
          stats.phone += 1;
          break;
        case 'CHIT':
          stats.chit += 1;
          break;
        case 'TEXTBOOK':
          stats.textbook += 1;
          break;
        case 'NOTEBOOK':
          stats.notebook += 1;
          break;
        case 'DEVICE':
          stats.device += 1;
          break;
      }
    });

    result.behaviors.forEach(behavior => {
      switch (behavior.type) {
        case 'HEAD_TURN':
        case 'LOOKING_AWAY':
          stats.headTurn += 1;
          break;
        case 'LOOKING_DOWN':
          stats.leaning += 1;
          break;
        case 'PROXIMITY_ALERT':
          stats.multiplePeople += 1;
          break;
      }
    });

    return stats;
  },

  convertDetectionsToAlerts: (result: AIDetection): Omit<ProctorAlert, 'id' | 'timestamp' | 'screenshot'>[] => {
    const alerts: Omit<ProctorAlert, 'id' | 'timestamp' | 'screenshot'>[] = [];

    result.prohibited_items.forEach(item => {
      const type = mapDetectionType(item);
      if (!type) return;

      const score = DETECTION_POINTS[type];
      alerts.push({
        type,
        seat: getSeatLabel(item.bbox),
        level: toAlertLevel(score),
        description: `${type.replace(/_/g, ' ')} detected.`,
        confidence: Math.max(0, Math.min(1, item.confidence ?? 0)),
        score,
      });
    });

    result.behaviors.forEach((behavior, index) => {
      const type = mapBehaviorType(behavior.type);
      if (!type) return;

      const score = DETECTION_POINTS[type];
      const behaviorBox = result.head_poses?.[index]?.bbox;
      alerts.push({
        type,
        seat: getSeatLabel(behaviorBox),
        level: toAlertLevel(score),
        description: `${behavior.type.replace(/_/g, ' ').toLowerCase()} detected.`,
        confidence: Math.max(0, Math.min(1, behavior.confidence ?? 0)),
        score,
      });
    });

    if (result.person_count === 0) {
      const score = DETECTION_POINTS.NO_PERSON;
      alerts.push({
        type: 'NO_PERSON',
        seat: 'S1',
        level: toAlertLevel(score),
        description: 'No student detected in the current frame.',
        confidence: 1,
        score,
      });
    }

    return alerts;
  },
};
