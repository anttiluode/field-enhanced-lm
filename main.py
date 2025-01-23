# main14.py

import argparse
import sys
import traceback
import torch
import transformers
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats, ndimage
import powerlaw
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import deque
import time
import logging
import cv2
import math
from threading import Thread
from transformers import TextIteratorStreamer

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Determine the device to use (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

@dataclass
class FieldState:
    """Represents the state of the critical field."""
    magnitude: torch.Tensor
    phase: torch.Tensor
    energy: float
    stability: float
    patterns: List[torch.Tensor]
    timestamp: float
    raw_frame: Optional[np.ndarray] = None  # For visual field

@dataclass
class PatternMetrics:
    """Metrics for detected patterns in the critical field."""
    power_law_exponent: float
    avalanche_sizes: List[float]
    correlation_length: float
    fractal_dimension: float
    complexity: float

class VisualFieldProcessor:
    """Processes webcam input and converts it into a field state."""
    def __init__(self, field_size=32, camera_index=0, device=device):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_index}")
            
        self.field_size = field_size
        self.device = device
        self.smoother = nn.AvgPool2d(2).to(self.device)
        self.last_frames = deque(maxlen=5)
        self.frame_history = deque(maxlen=100)  # For metrics
        
    def get_frame_field(self) -> FieldState:
        """
        Retrieves a webcam frame, converts it into a FieldState object,
        and ensures it's on the correct device with a small noise injection.
        """
        ret, frame = self.cap.read()
        if not ret:
            logging.warning("No frame read from camera. Returning a random visual field state.")
            
            # Return a random field so the visual field is never None.
            # This ensures there's always *some* energy in the visual field.
            random_magnitude = 0.1 * torch.randn(1, 1, self.field_size, self.field_size, dtype=torch.float32, device=self.device)
            random_phase = 2.0 * math.pi * torch.rand_like(random_magnitude, device=self.device)
            
            # Append to last_frames to ensure consistency
            self.last_frames.append(random_magnitude.clone())
            
            return FieldState(
                magnitude=random_magnitude,
                phase=random_phase,
                energy=torch.mean(random_magnitude**2).item(),
                stability=0.0,
                patterns=[],
                timestamp=time.time(),
                raw_frame=None
            )

        # If a frame is successfully read:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.field_size, self.field_size))
        normalized = torch.from_numpy(resized).float().div(255.0)
        
        # Unsqueeze to shape [1,1,H,W]
        magnitude = normalized.unsqueeze(0).unsqueeze(0).to(torch.float32).to(self.device)
        
        # Add a small noise term to avoid zero energy:
        magnitude += 0.01 * torch.randn_like(magnitude, device=self.device)
        
        phase = self._compute_phase(magnitude).to(torch.float32).to(self.device)
        
        # Store raw frame for visualization
        self.frame_history.append(frame)
        
        energy = torch.mean(magnitude**2).item()
        stability = self._compute_stability(magnitude)
        
        return FieldState(
            magnitude=magnitude,
            phase=phase,
            energy=energy,
            stability=stability,
            patterns=[],  # Patterns can be detected if needed
            timestamp=time.time(),
            raw_frame=frame
        )
        
    def _compute_phase(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Computes the phase based on the change in magnitude."""
        # Detach and clone to ensure tensors are on the correct device and no gradients are tracked
        self.last_frames.append(magnitude.detach().clone())
        if len(self.last_frames) > 1:
            diff = self.last_frames[-1] - self.last_frames[-2]
            phase = torch.atan2(diff, magnitude)
        else:
            phase = torch.rand_like(magnitude, device=self.device) * 2 * math.pi
        return phase.float()
        
    def _compute_stability(self, magnitude: torch.Tensor) -> float:
        """Computes the stability based on the change in magnitude."""
        if len(self.last_frames) > 1:
            last_magnitude = self.last_frames[-1].to(self.device)
            return torch.mean(torch.abs(magnitude - last_magnitude)).item()
        return 0.0
        
    def __del__(self):
        self.cap.release()

class CriticalField(nn.Module):
    """Neural field implementation maintaining critical dynamics."""
    def __init__(
        self,
        size: int = 32,
        coupling_strength: float = 0.015,
        dt: float = 0.05,
        device: torch.device = device,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.size = size
        self.coupling_strength = coupling_strength
        self.dt = dt
        self.device = device
        self.dtype = dtype
        
        # Field components with explicit dtype
        self.magnitude = nn.Parameter(torch.zeros(1, 1, size, size, dtype=dtype, device=device))
        self.phase = nn.Parameter(torch.zeros(1, 1, size, size, dtype=dtype, device=device))
        
        # Dynamic parameters with explicit dtype
        self.damping = nn.Parameter(torch.tensor(0.05, dtype=dtype, device=device))
        self.tension = nn.Parameter(torch.tensor(1.5, dtype=dtype, device=device))
        self.nonlinearity = nn.Parameter(torch.tensor(0.3, dtype=dtype, device=device))
        
        # Memory
        self.states = deque(maxlen=1000)
        
        # Initialize with scale-free noise
        self._initialize_field()
    
    def _initialize_field(self):
        """Initializes the field with scale-free noise."""
        # Create k-space frequencies on the correct device
        k = torch.fft.fftfreq(self.size, device=self.device).reshape(-1, 1)
        k = torch.sqrt(k * k + k.reshape(1, -1)**2)
        k[0, 0] = 1  # Avoid division by zero
        
        # Generate noise directly on the correct device
        noise = torch.randn(self.size, self.size, device=self.device)
        noise_fft = torch.fft.fft2(noise)
        scale_free = torch.fft.ifft2(noise_fft / k**0.75).real
        
        # Initialize field components
        self.magnitude.data = scale_free.unsqueeze(0).unsqueeze(0).float()
        self.phase.data = torch.rand_like(self.magnitude, device=self.device) * 2 * math.pi
        self.phase.data = self.phase.data.float()
    
    def _compute_laplacian(self, field: torch.Tensor) -> torch.Tensor:
        """Computes the Laplacian of the given field."""
        padded = F.pad(field, (1, 1, 1, 1), mode='circular')
        return (
            padded[:, :, :-2, 1:-1] +
            padded[:, :, 2:, 1:-1] +
            padded[:, :, 1:-1, :-2] +
            padded[:, :, 1:-1, 2:] -
            4 * field
        )
    
    def evolve(self) -> FieldState:
        """Evolves the field based on its dynamics and returns the new state."""
        # Compute dynamics
        magnitude_lap = self._compute_laplacian(self.magnitude)
        phase_lap = self._compute_laplacian(self.phase)
        
        # Update magnitude
        d_magnitude = (
            self.tension * magnitude_lap -
            self.damping * self.magnitude -
            self.nonlinearity * self.magnitude**3
        )
        
        # Update phase with coupling
        d_phase = (
            self.tension * phase_lap +
            self.coupling_strength * torch.sin(
                torch.roll(self.phase, 1, dims=2) - self.phase
            )
        )
        
        # Euler integration
        self.magnitude.data += self.dt * d_magnitude
        self.phase.data += self.dt * d_phase

        # Handle NaNs by reinitializing
        if torch.isnan(self.magnitude).any() or torch.isnan(self.phase).any():
            logging.warning("NaN detected in field, reinitializing.")
            self._initialize_field()
        
        # Clamp values to maintain stability
        self.magnitude.data.clamp_(-2.0, 2.0)
        self.phase.data.clamp_(0, 2 * math.pi)
        
        # Create state
        state = FieldState(
            magnitude=self.magnitude.detach(),
            phase=self.phase.detach(),
            energy=torch.mean(self.magnitude**2).item(),
            stability=torch.mean(torch.abs(d_magnitude)).item(),
            patterns=self._detect_patterns(),
            timestamp=time.time()
        )
        
        # Store state
        self.states.append(state)
        
        return state
    
    def _detect_patterns(self) -> List[torch.Tensor]:
        """Detects emergent patterns in the field using adaptive thresholding."""
        try:
            # Get magnitude and detach
            activation = torch.abs(self.magnitude).detach()
            if torch.isnan(activation).any():
                logging.warning("NaN values detected in field magnitude")
                return []
                
            # Normalize activation to [0,1] range
            activation = (activation - activation.min()) / (activation.max() - activation.min() + 1e-8)
            
            # Adaptive thresholding
            mean = activation.mean()
            std = activation.std()
            if torch.isnan(std) or std < 1e-8:
                logging.warning("Invalid statistics in field activation")
                return []
                
            # Use lower threshold to catch more patterns
            threshold = mean + 1.5 * std
            binary = (activation > threshold).float()
            
            # Safety check on binary mask
            if binary.sum() < 1:
                return []
            
            # Find connected components
            try:
                binary_np = binary.cpu().numpy()[0, 0]
                labeled, num_features = ndimage.label(binary_np)
            except Exception as e:
                logging.error(f"Error in connected components: {e}")
                return []
                
            patterns = []
            max_patterns = 50  # Limit number of patterns
            
            # Extract patterns with size filtering
            min_size = 4  # Minimum pattern size
            for i in range(1, min(num_features + 1, max_patterns)):
                try:
                    # Create pattern mask
                    pattern_mask = torch.from_numpy(labeled == i).float().to(self.device)
                    
                    # Apply mask to activation and detach
                    pattern = (pattern_mask * activation[0, 0]).detach()
                    
                    # Filter by size and energy
                    pattern_size = pattern_mask.sum()
                    pattern_energy = pattern.sum()
                    
                    if pattern_size >= min_size and pattern_energy > 0:
                        # Normalize pattern
                        pattern = pattern / (pattern.max() + 1e-8)
                        patterns.append(pattern)
                        
                except Exception as e:
                    logging.error(f"Error processing pattern {i}: {e}")
                    continue
                    
            logging.info(f"Detected {len(patterns)} valid patterns")
            return patterns
                
        except Exception as e:
            logging.error(f"Pattern detection failed: {e}")
            return []

class CriticalPatternAnalyzer:
    """Analyzes emergent patterns and criticality in the field."""
    def __init__(self):
        self.expected_critical_exponent = -1.5
        self.correlation_threshold = 0.7
        self.min_avalanche_size = 3
        self.pattern_history = []
        self.max_history = 1000
    
    def analyze_field(self, field_state: FieldState) -> Optional[PatternMetrics]:
        """Analyzes the given field state and returns pattern metrics if applicable."""
        field = field_state.magnitude.cpu().numpy()
        
        # Detect avalanches
        threshold = np.mean(field) + 2 * np.std(field)
        avalanches = self._detect_avalanches(field > threshold)
        avalanche_sizes = [np.sum(avalanche) for avalanche in avalanches]
        
        if len(avalanche_sizes) < self.min_avalanche_size:
            return None
        
        # Fit power law
        try:
            fit = powerlaw.Fit(avalanche_sizes, discrete=True)
            power_law_exponent = fit.power_law.alpha
        except Exception as e:
            logging.error(f"Power law fitting failed: {e}")
            power_law_exponent = np.nan
        
        # Calculate metrics
        correlation_length = self._compute_correlation_length(field)
        fractal_dim = self._estimate_fractal_dimension(field > threshold)
        complexity = self._compute_complexity(field)
        
        metrics = PatternMetrics(
            power_law_exponent=power_law_exponent,
            avalanche_sizes=avalanche_sizes,
            correlation_length=correlation_length,
            fractal_dimension=fractal_dim,
            complexity=complexity
        )
        
        # Store in history
        self.pattern_history.append({
            'metrics': metrics,
            'timestamp': field_state.timestamp
        })
        if len(self.pattern_history) > self.max_history:
            self.pattern_history.pop(0)
        
        return metrics
    
    def _detect_avalanches(self, binary_field: np.ndarray) -> List[np.ndarray]:
        """Detects avalanches in the binary field."""
        labeled, num_features = ndimage.label(binary_field)
        return [labeled == i for i in range(1, num_features + 1)]
    
    def _compute_correlation_length(self, field: np.ndarray) -> float:
        """Computes the correlation length of the field."""
        corr = np.zeros(min(field.shape) // 2)
        for r in range(len(corr)):
            rolled = np.roll(field, r, axis=0)
            corr[r] = np.mean(field * rolled)
        
        x = np.arange(len(corr))
        y = np.log(corr + 1e-10)
        slope, _, _, _, _ = stats.linregress(x[:len(corr)//2], y[:len(corr)//2])
        
        return -1/slope if slope < 0 else np.inf
    
    def _estimate_fractal_dimension(self, binary_field: np.ndarray) -> float:
        """Estimates the fractal dimension of the binary field using box counting."""
        def count_boxes(field, box_size):
            try:
                shape = field.shape
                # Ensure divisible by box_size
                new_shape = (shape[0] // box_size * box_size, shape[1] // box_size * box_size)
                field_cropped = field[:new_shape[0], :new_shape[1]]
                blocks = field_cropped.reshape(
                    new_shape[0]//box_size, box_size,
                    new_shape[1]//box_size, box_size
                ).any(axis=(1,3))
                return np.sum(blocks)
            except Exception as e:
                logging.error(f"Error in box counting: {e}")
                return 0
        
        box_sizes = [2**i for i in range(1, int(np.log2(min(binary_field.shape))) + 1)]
        counts = [count_boxes(binary_field, size) for size in box_sizes if size <= min(binary_field.shape)]
        
        if len(counts) < 2:
            return 0.0
        
        x = np.log(box_sizes[:len(counts)])
        y = np.log(counts)
        slope, _, _, _, _ = stats.linregress(x, y)
        
        return -slope
    
    def _compute_complexity(self, field: np.ndarray) -> float:
        """Computes the complexity of the field based on integration and differentiation."""
        # Integration: global correlations
        try:
            global_corr = np.corrcoef(field.reshape(-1, field.shape[-1]))
            integration = np.mean(np.abs(global_corr))
        except Exception as e:
            logging.error(f"Error computing global correlation: {e}")
            integration = 0.0
        
        # Differentiation: local variability
        local_var = np.std(field)
        
        # Complexity peaks when balanced
        return 4 * integration * (1 - integration) * local_var
    
    def is_critical(self, metrics: PatternMetrics) -> bool:
        """Determines if the field is in a critical state based on multiple indicators."""
        # Check multiple indicators
        exponent_match = abs(metrics.power_law_exponent - self.expected_critical_exponent) < 0.3
        recent_correlations = [m['metrics'].correlation_length for m in self.pattern_history[-100:]]
        avg_correlation = np.mean(recent_correlations) if recent_correlations else 0.0
        long_correlations = metrics.correlation_length > avg_correlation
        recent_complexities = [m['metrics'].complexity for m in self.pattern_history[-100:]]
        avg_complexity = np.mean(recent_complexities) if recent_complexities else 0.0
        high_complexity = metrics.complexity > avg_complexity
        fractal_ok = 1.1 < metrics.fractal_dimension < 1.9
        
        return sum([exponent_match, long_correlations, high_complexity, fractal_ok]) >= 3
    
    def get_temporal_metrics(self) -> Dict:
        """Returns temporal metrics based on the pattern history."""
        if len(self.pattern_history) < 2:
            return {}
        
        exponents = [m['metrics'].power_law_exponent for m in self.pattern_history]
        complexities = [m['metrics'].complexity for m in self.pattern_history]
        
        return {
            'exponent_stability': np.std(exponents),
            'complexity_trend': np.polyfit(range(len(complexities)), complexities, 1)[0],
            'pattern_diversity': len(set(tuple(s.avalanche_sizes) for s in self.pattern_history[-100:]))
        }

class PredictiveCrossModalCoupling(nn.Module):
    """Enhanced cross-modal coupling with predictive mechanisms"""
    def __init__(self, field_size=32, hidden_size=2560, device=device):
        super().__init__()
        self.field_size = field_size
        self.device = device
        self.hidden_size = hidden_size
        
        # Prediction networks
        # Updated to hidden_size=2560
        self.visual_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)  # Predict both magnitude and phase (hidden_size)
        ).to(device)
        
        self.language_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)  # Predict both magnitude and phase (hidden_size)
        ).to(device)
        
        # Cross-modal attention layers with embed_dim=2560
        self.visual_to_language = nn.MultiheadAttention(
            embed_dim=hidden_size,  # Updated from 1024 to 2560
            num_heads=8,            # Adjusted number of heads accordingly
            batch_first=True
        ).to(device)
        
        self.language_to_visual = nn.MultiheadAttention(
            embed_dim=hidden_size,  # Updated from 1024 to 2560
            num_heads=8,            # Adjusted number of heads accordingly
            batch_first=True
        ).to(device)
        
        # Field encoders (float32)
        self.visual_encoder = self._create_encoder(field_size)
        self.language_encoder = self._create_encoder(field_size)
        
        # Prediction history for analysis
        self.prediction_history = deque(maxlen=1000)
        
    def _create_encoder(self, field_size):
        """Creates an encoder for field states."""
        return nn.Sequential(
            nn.Linear(2 * field_size * field_size, self.hidden_size),  # Project to hidden_size
            nn.GELU(),
            nn.LayerNorm(self.hidden_size)
        ).to(self.device)
        
    def _encode_field_state(self, state: FieldState) -> torch.Tensor:
        """Encodes a field state using the provided encoder."""
        if state is None:
            return None
        # Combine magnitude and phase
        combined = torch.cat([
            state.magnitude.flatten(),
            state.phase.flatten()
        ]).to(self.device)
        return self.visual_encoder(combined).unsqueeze(0)  # Shape: [1, 1, hidden_size]
        
    def _compute_prediction_error(self, predicted: torch.Tensor, actual: torch.Tensor) -> float:
        """Compute normalized prediction error"""
        if predicted is None or actual is None:
            return 1.0  # Maximum error when prediction is impossible
            
        error = F.mse_loss(predicted, actual)
        return torch.clamp(error, 0.0, 1.0).item()
        
    def forward(self, visual_state: Optional[FieldState], language_state: FieldState):
        """Performs bidirectional coupling with predictive mechanisms."""
        # Encode states
        vis_encoded = self._encode_field_state(visual_state) if visual_state else None  # [1,1,hidden_size]
        lang_encoded = self._encode_field_state(language_state)  # [1,1,hidden_size]
        
        # Make predictions if both states are available
        if vis_encoded is not None and lang_encoded is not None:
            # Predict next states
            predicted_visual = self.visual_predictor(lang_encoded.squeeze(0))  # [1, hidden_size]
            predicted_language = self.language_predictor(vis_encoded.squeeze(0))  # [1, hidden_size]
            
            # Compute prediction errors
            visual_error = self._compute_prediction_error(predicted_visual, lang_encoded.squeeze(0))
            language_error = self._compute_prediction_error(predicted_language, vis_encoded.squeeze(0))
            
            # Dynamic coupling strength based on prediction accuracy
            coupling_strength = 1.0 / (1.0 + visual_error + language_error)
            
            # Cross attention with dynamic strength
            lang_to_vis, lang_weights = self.language_to_visual(
                query=vis_encoded,    # [1,1,hidden_size]
                key=lang_encoded,     # [1,1,hidden_size]
                value=lang_encoded    # [1,1,hidden_size]
            )
            
            vis_to_lang, vis_weights = self.visual_to_language(
                query=lang_encoded,    # [1,1,hidden_size]
                key=vis_encoded,      # [1,1,hidden_size]
                value=vis_encoded     # [1,1,hidden_size]
            )
            
            # Scale influences by coupling strength
            lang_to_vis = lang_to_vis * coupling_strength  # [1,1,hidden_size]
            vis_to_lang = vis_to_lang * coupling_strength  # [1,1,hidden_size]
            
        else:
            lang_to_vis = torch.zeros(1, 1, self.hidden_size, device=self.device)
            vis_to_lang = torch.zeros(1, 1, self.hidden_size, device=self.device)
            visual_error = 1.0
            language_error = 1.0
            coupling_strength = 0.0
            lang_weights = torch.zeros(1, 1, 1, device=self.device)
            vis_weights = torch.zeros(1, 1, 1, device=self.device)
        
        # Store metrics
        metrics = {
            'visual_prediction_error': visual_error,
            'language_prediction_error': language_error,
            'coupling_strength': coupling_strength,
            'lang_attention_mean': lang_weights.mean().item(),
            'vis_attention_mean': vis_weights.mean().item(),
            'timestamp': time.time()
        }
        
        self.prediction_history.append(metrics)
        
        return lang_to_vis, vis_to_lang, metrics
        
    def get_prediction_metrics(self) -> Dict:
        """Returns averaged prediction metrics from recent history"""
        if not self.prediction_history:
            return {}
            
        recent = list(self.prediction_history)[-100:]
        return {
            'avg_visual_error': np.mean([x['visual_prediction_error'] for x in recent]),
            'avg_language_error': np.mean([x['language_prediction_error'] for x in recent]),
            'avg_coupling_strength': np.mean([x['coupling_strength'] for x in recent]),
            'error_stability': np.std([x['visual_prediction_error'] + x['language_prediction_error'] for x in recent])
        }

class CriticalFieldLayer(nn.Module):
    """Layer that couples language and visual model hidden states with critical fields."""
    def __init__(
        self,
        hidden_size: int,
        field_size: int = 32,
        coupling_strength: float = 0.1,
        device: torch.device = device
    ):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.field_size = field_size
        
        # Projection layers to map between model hidden size and field embedding size
        self.proj_up = nn.Linear(hidden_size, hidden_size).to(self.device)    # hidden_size -> hidden_size
        self.proj_down = nn.Linear(hidden_size, hidden_size).to(self.device)  # hidden_size -> hidden_size
        
        # Critical fields
        self.language_field = CriticalField(
            size=field_size,
            coupling_strength=coupling_strength,
            device=self.device
        )
        self.visual_field = CriticalField(
            size=field_size,
            coupling_strength=coupling_strength,
            device=self.device
        )
        
        # Predictive coupling
        self.field_coupling = PredictiveCrossModalCoupling(
            field_size=field_size,
            hidden_size=hidden_size,  # Ensure hidden_size matches model's hidden_size
            device=self.device
        )
        
        # Projections (float32)
        self.to_field = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, dtype=torch.float32, device=self.device),
            nn.GELU(),
            nn.Linear(hidden_size, 2 * field_size * field_size, dtype=torch.float32, device=self.device)  # 2 channels for magnitude and phase
        )
        
        self.from_field = nn.Sequential(
            nn.Linear(field_size * field_size * 2, hidden_size, dtype=torch.float32, device=self.device),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, dtype=torch.float32, device=self.device)
        )
        
        # History tracking
        self.state_history = deque(maxlen=1000)
        
        # Move all components to device
        self.to(self.device)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        visual_state: Optional[FieldState],
        language_state: FieldState
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Couples hidden states with critical fields and returns enhanced hidden states
        along with field information.
        """
        # Ensure hidden_states is float32 and on device
        hidden_states = hidden_states.to(torch.float32).to(self.device)
        
        # Project hidden states up to hidden_size
        projected_hidden = self.proj_up(hidden_states)  # [batch, hidden_size] -> [batch, hidden_size]
        
        # Project to field perturbations
        field_proj = self.to_field(hidden_states)  # [batch, hidden_size] -> [batch, 2*field_size*field_size]
        
        field_proj = field_proj.view(-1, 2, self.field_size, self.field_size)  # [batch, 2, 32, 32]
        
        # Split into magnitude and phase perturbations
        magnitude_pert = field_proj[:, 0].mean(dim=0, keepdim=True)  # [1, 32, 32]
        phase_pert = field_proj[:, 1].mean(dim=0, keepdim=True)      # [1, 32, 32]
        
        # Apply perturbations to language field
        self.language_field.magnitude += magnitude_pert
        self.language_field.phase += phase_pert
        
        # Evolve fields
        evolved_language = self.language_field.evolve()
        evolved_visual = self.visual_field.evolve() if visual_state else None
        
        # Get predictive coupling influences
        lang_to_vis_influence, vis_to_lang_influence, coupling_metrics = self.field_coupling(
            visual_state=evolved_visual,
            language_state=evolved_language
        )
        
        # Process field states
        enhanced_language = self.from_field(
            torch.cat([
                evolved_language.magnitude.flatten(),
                evolved_language.phase.flatten()
            ], dim=0).to(self.device)  # [2*32*32=2048]
        )
        
        if evolved_visual:
            enhanced_visual = self.from_field(
                torch.cat([
                    evolved_visual.magnitude.flatten(),
                    evolved_visual.phase.flatten()
                ], dim=0).to(self.device)  # [2*32*32=2048]
            )
        else:
            enhanced_visual = torch.zeros_like(enhanced_language, device=self.device)
        
        # -------------------------------------------------------------------
        # Project coupling influences from hidden_size to hidden_size
        # -------------------------------------------------------------------
        lang_to_vis_proj = self.proj_down(lang_to_vis_influence)   # shape => [1, hidden_size]
        vis_to_lang_proj = self.proj_down(vis_to_lang_influence)   # shape => [1, hidden_size]
        
        # -------------------------------------------------------------------
        # Combine everything with dynamic coupling
        # -------------------------------------------------------------------
        coupling_strength = coupling_metrics['coupling_strength']
        enhanced = (
            enhanced_language +
            coupling_strength * (
                0.1 * enhanced_language +
                0.05 * enhanced_visual +
                0.1 * lang_to_vis_proj.squeeze(0) +
                0.05 * vis_to_lang_proj.squeeze(0)
            )
        )
        
        # Update metrics with prediction information
        field_info = {
            'language_energy': evolved_language.energy,
            'language_stability': evolved_language.stability,
            'visual_energy': evolved_visual.energy if evolved_visual else 0.0,
            'visual_stability': evolved_visual.stability if evolved_visual else 0.0,
            'coupling_metrics': coupling_metrics,
            'prediction_metrics': self.field_coupling.get_prediction_metrics(),
            'timestamp': time.time()
        }
        
        # Store state history
        self.state_history.append({
            'field_info': field_info,
            'coupling_metrics': coupling_metrics,
            'timestamp': time.time()
        })
        
        return enhanced, field_info
        
    def get_state_history(self) -> List[Dict]:
        """Returns the history of field states and metrics"""
        return list(self.state_history)

class CriticalFieldPhi:
    """Main class integrating Phi-2 with critical field layers and webcam input."""
    def __init__(self, camera_index=0):
        # Check or define device
        self.device = device

        # Load Phi model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/phi-2",
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            trust_remote_code=True,
            torch_dtype=torch.float32  # Changed from float16 to float32 for consistency
        ).to(self.device)
        
        # Configure tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Add special tokens
        special_tokens = {
            'field_prefix': '<|field|>',
            'field_memory': '<|field:memory|>',
            'field_pattern': '<|field:pattern|>'
        }
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': list(special_tokens.values())
        })
        
        # Resize model embeddings
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Initialize critical field layers
        hidden_size = self.model.config.hidden_size  # e.g., 2560
        logging.info(f"Model Hidden Size: {hidden_size}")
        self.field_layer = CriticalFieldLayer(
            hidden_size=hidden_size,
            field_size=32,
            coupling_strength=0.1,
            device=self.device
        ).to(self.device)
        
        # Initialize visual field processor
        self.visual_processor = VisualFieldProcessor(camera_index=camera_index, device=self.device)
        
        # Initialize pattern analyzer
        self.pattern_analyzer = CriticalPatternAnalyzer()
        
        # History tracking
        self.history = []
        self.current_field_info = None
        
        # Patch model's forward method
        self._patch_forward_method()
        
        logging.info("Initialized CriticalFieldPhi with Dual Fields (Bidirectional)")
    
    def _patch_forward_method(self):
        """Patches the model's forward method to include field dynamics."""
        original_forward = self.model.forward
        
        def forward_with_dual_fields(*args, **kwargs):
            # Ensure we get hidden states
            kwargs["output_hidden_states"] = True
            
            # Call original forward
            outputs = original_forward(*args, **kwargs)
            
            # Safety check for hidden states
            if not hasattr(outputs, 'hidden_states'):
                logging.warning("No hidden states in output, creating default hidden state")
                hidden_state = torch.zeros(1, self.hidden_size, device=self.device)
            else:
                hidden_state = outputs.hidden_states[-1]
            
            # Evolve fields
            evolved_visual = self.field_layer.visual_field.evolve()
            language_state = self.field_layer.language_field.evolve()
            
            # Predictive coupling with safety checks
            enhanced_hidden, field_info = self.field_layer(
                hidden_states=hidden_state,
                visual_state=evolved_visual,
                language_state=language_state
            )
            
            self.current_field_info = field_info
            
            # Update hidden states
            if hasattr(outputs, 'hidden_states'):
                outputs.hidden_states = tuple(
                    list(outputs.hidden_states[:-1]) + [enhanced_hidden]
                )
            
            return outputs
        
        self.model.forward = forward_with_dual_fields
    
    def generate_response(
        self,
        text: str,
        max_length: int = 200,
        temperature: float = 0.7,
        field_params: Dict = {}
    ) -> Tuple[str, Dict]:
        """
        Generates a response from the language model, optionally performing field analysis.
        field_params: Dictionary containing field parameters to adjust.
        """
        # Update field parameters if provided
        if field_params:
            self.update_field_parameters(field_params)
        
        # Check for field queries
        field_query = False
        if '<|field|>' in text:
            field_query = True
            text = text.replace('<|field|>', '')
        
        # Clean input
        text = text.strip()
        if not text:
            text = "Hello"
        
        # Tokenize input with attention mask
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        ).to(self.device)
        
        # Create the streamer
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_special_tokens=True,
            legacy_cache_format=False  # Add this to handle new cache format
        )

        # Set up generation kwargs
        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": max_length,       # Control total new tokens
            "temperature": temperature,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "do_sample": True,
            "return_legacy_cache": True,        # Handle cache format
            "streamer": streamer,
            "pad_token_id": self.tokenizer.eos_token_id,
            "output_hidden_states": True,       # Ensure we get hidden states
            "return_dict_in_generate": True     # Get structured output
        }
        
        # Launch generation in a separate thread
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Now read tokens as they arrive
        partial_tokens = []
        for new_text in streamer:
            partial_tokens.append(new_text)
            # If you want to see partial output in console, just print it:
            print(new_text, end="", flush=True)
        
        # Join the partial tokens into the final string
        response = "".join(partial_tokens)
        
        # Analyze field if requested
        if field_query:
            # Evolve field multiple times for better dynamics
            for _ in range(3):
                field_state = self.field_layer.language_field.evolve()
            
            metrics = self.pattern_analyzer.analyze_field(field_state)
            
            if metrics:
                analysis = (
                    f"\n\n[Field Analysis]"
                    f"\nPower Law Exponent: {metrics.power_law_exponent:.3f}"
                    f"\nCorrelation Length: {metrics.correlation_length:.3f}"
                    f"\nComplexity: {metrics.complexity:.3f}"
                    f"\nCritical State: {self.pattern_analyzer.is_critical(metrics)}"
                )
                response += analysis
                
                # Update field info with more detailed metrics
                self.current_field_info = {
                    'language_energy': field_state.energy,
                    'language_stability': field_state.stability,
                    'visual_energy': field_state.energy,  # Assuming visual energy similar
                    'visual_stability': field_state.stability,  # Assuming visual stability similar
                    'coupling_metrics': self.field_layer.field_coupling.get_prediction_metrics(),
                    'timestamp': time.time()
                }
        
        # Store interaction
        self.history.append({
            'input': text,
            'response': response,
            'field_info': self.current_field_info,
            'timestamp': time.time()
        })
        
        return response, self.current_field_info
    
    def update_field_parameters(self, params: Dict):
        """Updates field parameters based on user input from sliders."""
def update_field_parameters(self, params: Dict):
    """Updates field parameters based on user input from sliders."""
    # Update language field parameters
    if 'language_coupling_strength' in params:
        self.field_layer.language_field.coupling_strength = params['language_coupling_strength']
        logging.info(f"Updated language_coupling_strength to {params['language_coupling_strength']}")
    
    if 'language_damping' in params:
        # Correct way to update nn.Parameter
        self.field_layer.language_field.damping.data = torch.tensor(params['language_damping'], device=self.device)
        logging.info(f"Updated language_damping to {params['language_damping']}")
    
    if 'language_tension' in params:
        self.field_layer.language_field.tension.data = torch.tensor(params['language_tension'], device=self.device)
        logging.info(f"Updated language_tension to {params['language_tension']}")
    
    if 'language_nonlinearity' in params:
        self.field_layer.language_field.nonlinearity.data = torch.tensor(params['language_nonlinearity'], device=self.device)
        logging.info(f"Updated language_nonlinearity to {params['language_nonlinearity']}")
        # Update visual field parameters
        if 'visual_coupling_strength' in params:
            self.field_layer.visual_field.coupling_strength = params['visual_coupling_strength']
            logging.info(f"Updated visual_coupling_strength to {params['visual_coupling_strength']}")
        
        if 'visual_damping' in params:
            self.field_layer.visual_field.damping = torch.tensor(params['visual_damping'], device=self.device)
            logging.info(f"Updated visual_damping to {params['visual_damping']}")
        
        if 'visual_tension' in params:
            self.field_layer.visual_field.tension = torch.tensor(params['visual_tension'], device=self.device)
            logging.info(f"Updated visual_tension to {params['visual_tension']}")
        
        if 'visual_nonlinearity' in params:
            self.field_layer.visual_field.nonlinearity = torch.tensor(params['visual_nonlinearity'], device=self.device)
            logging.info(f"Updated visual_nonlinearity to {params['visual_nonlinearity']}")
        
        # Update coupling strength in PredictiveCrossModalCoupling
        if 'cross_coupling_strength' in params:
            self.field_layer.field_coupling.coupling_strength = params['cross_coupling_strength']
            logging.info(f"Updated cross_coupling_strength to {params['cross_coupling_strength']}")

def create_dual_field_plot(visual_state: Optional[FieldState], language_state: FieldState, metrics: Dict) -> Optional[go.Figure]:
    """Create enhanced visualization including prediction metrics"""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Visual Field Magnitude', 'Language Field Magnitude',
            'Visual Field Phase', 'Language Field Phase',
            'Prediction Errors & Coupling', 'Field Metrics'
        ),
        specs=[
            [{'type': 'heatmap'}, {'type': 'heatmap'}],
            [{'type': 'heatmap'}, {'type': 'heatmap'}],
            [{'type': 'scatter'}, {'type': 'bar'}]
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # Visual field magnitude
    if visual_state:
        v_magnitude = visual_state.magnitude.cpu().numpy()[0, 0]
        v_magnitude = (v_magnitude - v_magnitude.min()) / (v_magnitude.max() - v_magnitude.min() + 1e-8)
        fig.add_trace(
            go.Heatmap(z=v_magnitude, colorscale='Viridis', showscale=True),
            row=1, col=1
        )
        
        # Visual field phase
        v_phase = visual_state.phase.cpu().numpy()[0, 0]
        v_phase = (v_phase + np.pi) / (2 * np.pi)
        fig.add_trace(
            go.Heatmap(z=v_phase, colorscale='RdBu', showscale=True),
            row=2, col=1
        )
    else:
        fig.add_trace(go.Scatter(), row=1, col=1)
        fig.add_trace(go.Scatter(), row=2, col=1)
    
    # Language field magnitude
    l_magnitude = language_state.magnitude.cpu().numpy()[0, 0]
    l_magnitude = (l_magnitude - l_magnitude.min()) / (l_magnitude.max() - l_magnitude.min() + 1e-8)
    fig.add_trace(
        go.Heatmap(z=l_magnitude, colorscale='Viridis', showscale=True),
        row=1, col=2
    )
    
    # Language field phase
    l_phase = language_state.phase.cpu().numpy()[0, 0]
    l_phase = (l_phase + np.pi) / (2 * np.pi)
    fig.add_trace(
        go.Heatmap(z=l_phase, colorscale='RdBu', showscale=True),
        row=2, col=2
    )
    
    # Prediction errors and coupling strength
    prediction_metrics = metrics.get('prediction_metrics', {})
    coupling_metrics = metrics.get('coupling_metrics', {})
    
    if prediction_metrics and coupling_metrics:
        # Create time axis
        time_window = 100  # Show last 100 timesteps
        x_time = list(range(time_window))
        
        # Prediction errors
        visual_errors = [prediction_metrics.get('avg_visual_error', 0)] * time_window
        language_errors = [prediction_metrics.get('avg_language_error', 0)] * time_window
        coupling_strengths = [coupling_metrics.get('avg_coupling_strength', 0)] * time_window
        
        # Plot error curves
        fig.add_trace(
            go.Scatter(
                x=x_time,
                y=visual_errors,
                name='Visual Error',
                line=dict(color='rgba(0, 255, 0, 0.8)'),
                fill='tozeroy'
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=x_time,
                y=language_errors,
                name='Language Error',
                line=dict(color='rgba(255, 0, 0, 0.8)'),
                fill='tozeroy'
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=x_time,
                y=coupling_strengths,
                name='Coupling Strength',
                line=dict(color='rgba(0, 0, 255, 0.8)'),
                fill='tozeroy'
            ),
            row=3, col=1
        )
        
        # Update subplot settings
        fig.update_xaxes(title_text='Time Steps', row=3, col=1)
        fig.update_yaxes(title_text='Error / Strength', row=3, col=1)
    
    # Field metrics as bar chart
    metrics_to_plot = {
        'Language Energy': metrics.get('language_energy', 0),
        'Language Stability': metrics.get('language_stability', 0),
        'Visual Energy': metrics.get('visual_energy', 0),
        'Visual Stability': metrics.get('visual_stability', 0),
        'Prediction Accuracy': 1.0 - (prediction_metrics.get('avg_visual_error', 0) + 
                                    prediction_metrics.get('avg_language_error', 0)) / 2
    }
    
    fig.add_trace(
        go.Bar(
            x=list(metrics_to_plot.keys()),
            y=list(metrics_to_plot.values()),
            marker_color=['blue', 'green', 'red', 'purple', 'orange']
        ),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=1200,
        width=1000,
        showlegend=True,
        title_text="Dual Critical Field State Visualization (Predictive)",
        template="plotly_dark",
        barmode='group'
    )
    
    # Adjust bar chart axes
    fig.update_xaxes(tickangle=45, row=3, col=2)
    
    return fig

def format_metrics_text(metrics: Dict) -> str:
    """Format all metrics for display, including prediction metrics"""
    prediction_metrics = metrics.get('prediction_metrics', {})
    coupling_metrics = metrics.get('coupling_metrics', {})
    
    basic_metrics = (
        f"Language Energy: {metrics.get('language_energy', 0):.3f}\n"
        f"Language Stability: {metrics.get('language_stability', 0):.3f}\n"
        f"Visual Energy: {metrics.get('visual_energy', 0):.3f}\n"
        f"Visual Stability: {metrics.get('visual_stability', 0):.3f}\n"
    )
    
    prediction_info = (
        f"\nPrediction Metrics:\n"
        f"Visual Error: {prediction_metrics.get('avg_visual_error', 0):.3f}\n"
        f"Language Error: {prediction_metrics.get('avg_language_error', 0):.3f}\n"
        f"Error Stability: {prediction_metrics.get('error_stability', 0):.3f}\n"
    )
    
    coupling_info = (
        f"\nCoupling Metrics:\n"
        f"Coupling Strength: {coupling_metrics.get('avg_coupling_strength', 0):.3f}\n"
        f"Visual→Language: {coupling_metrics.get('vis_attention_mean', 0):.3f}\n"
        f"Language→Visual: {coupling_metrics.get('lang_attention_mean', 0):.3f}\n"
    )
    
    timestamp = metrics.get('timestamp', time.time())
    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
    
    return (
        f"Field State Metrics:\n"
        f"{basic_metrics}"
        f"{prediction_info}"
        f"{coupling_info}"
        f"\nTimestamp: {time_str}"
    )

def create_interface(phi: CriticalFieldPhi):
    """Creates the Gradio interface for the dual field system with bidirectional coupling."""
    with gr.Blocks(title="Dual Field Enhanced Phi (Bidirectional)") as interface:
        gr.Markdown("# Dual Field Enhanced Phi (Bidirectional)")
        
        with gr.Row():
            # Input column
            with gr.Column(scale=1):
                input_text = gr.Textbox(
                    label="Your message",
                    placeholder="Use <|field|> for field analysis"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    label="Temperature"
                )
                
                # Sliders for Field Parameters
                with gr.Accordion("Field Parameters", open=False):
                    gr.Markdown("### Language Field Parameters")
                    language_coupling_strength = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.1,
                        label="Language Coupling Strength"
                    )
                    language_damping = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.05,
                        label="Language Damping"
                    )
                    language_tension = gr.Slider(
                        minimum=0.0,
                        maximum=5.0,
                        step=0.1,
                        value=1.5,
                        label="Language Tension"
                    )
                    language_nonlinearity = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.3,
                        label="Language Nonlinearity"
                    )
                    
                    gr.Markdown("### Visual Field Parameters")
                    visual_coupling_strength = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.1,
                        label="Visual Coupling Strength"
                    )
                    visual_damping = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.05,
                        label="Visual Damping"
                    )
                    visual_tension = gr.Slider(
                        minimum=0.0,
                        maximum=5.0,
                        step=0.1,
                        value=1.5,
                        label="Visual Tension"
                    )
                    visual_nonlinearity = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.3,
                        label="Visual Nonlinearity"
                    )
                    
                    gr.Markdown("### Cross-Modal Coupling Parameters")
                    cross_coupling_strength = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.1,
                        label="Cross-Coupling Strength"
                    )
                
                response_text = gr.Textbox(
                    label="Response",
                    lines=10
                )
            
            # Visualization column
            with gr.Column(scale=2):
                field_plot = gr.Plot(label="Field States")
                
                with gr.Row():
                    webcam_preview = gr.Image(label="Webcam Input")
                    metrics_text = gr.Textbox(
                        label="Field Metrics",
                        lines=8
                    )
        
        def process_input(message: str, temp: float,
                         lang_coupling: float, lang_damping: float, lang_tension: float, lang_nonlinearity: float,
                         vis_coupling: float, vis_damping: float, vis_tension: float, vis_nonlinearity: float,
                         cross_coupling: float):
            """Processes user input, generates response, and updates visualizations."""
            try:
                # Collect field parameters
                field_params = {
                    'language_coupling_strength': lang_coupling,
                    'language_damping': lang_damping,
                    'language_tension': lang_tension,
                    'language_nonlinearity': lang_nonlinearity,
                    'visual_coupling_strength': vis_coupling,
                    'visual_damping': vis_damping,
                    'visual_tension': vis_tension,
                    'visual_nonlinearity': vis_nonlinearity,
                    'cross_coupling_strength': cross_coupling
                }
                
                # Generate response
                response, field_info = phi.generate_response(
                    message,
                    temperature=temp,
                    field_params=field_params
                )
                
                # Get latest field states
                visual_state = phi.visual_processor.get_frame_field()
                language_state = phi.field_layer.language_field.evolve()
                
                # Create visualizations
                fig = create_dual_field_plot(
                    visual_state=visual_state,
                    language_state=language_state,
                    metrics=field_info if field_info else {}
                )
                
                # Get webcam preview
                if visual_state and visual_state.raw_frame is not None:
                    webcam_image = cv2.cvtColor(visual_state.raw_frame, cv2.COLOR_BGR2RGB)
                    webcam_preview_image = webcam_image
                else:
                    webcam_preview_image = None
                
                # Get metrics summary
                if field_info:
                    coupling_metrics = field_info.get('coupling_metrics', {})
                    prediction_metrics = field_info.get('prediction_metrics', {})
                    metrics_str = "\n".join(
                        f"{k}: {v:.3f}" for k, v in coupling_metrics.items()
                    )
                    prediction_str = "\n".join(
                        f"{k}: {v:.3f}" for k, v in prediction_metrics.items()
                    )
                    field_metrics = (
                        f"Language Energy: {field_info.get('language_energy', 0.0):.3f}\n"
                        f"Language Stability: {field_info.get('language_stability', 0.0):.3f}\n"
                        f"Visual Energy: {field_info.get('visual_energy', 0.0):.3f}\n"
                        f"Visual Stability: {field_info.get('visual_stability', 0.0):.3f}\n"
                        f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(field_info.get('timestamp', time.time())))}\n"
                        f"Coupling Metrics:\n{metrics_str}\n"
                        f"Prediction Metrics:\n{prediction_str}"
                    )
                else:
                    field_metrics = "No field analysis available"
                    fig = None
                    webcam_preview_image = None
                
                return (
                    response,
                    fig,
                    webcam_preview_image,
                    field_metrics
                )
            except Exception as e:
                logging.error(f"Error in process_input: {str(e)}")
                traceback.print_exc()
                return (
                    f"Error: {str(e)}",
                    None,
                    None,
                    "Error processing input"
                )
        
        # Connect interface elements
        input_text.submit(
            fn=process_input,
            inputs=[
                input_text, 
                temperature,
                language_coupling_strength, language_damping, language_tension, language_nonlinearity,
                visual_coupling_strength, visual_damping, visual_tension, visual_nonlinearity,
                cross_coupling_strength
            ],
            outputs=[response_text, field_plot, webcam_preview, metrics_text]
        )
        
        # Optionally, add a refresh button or periodic update for webcam preview
        # Currently, webcam preview updates only on input submission
        
        return interface

def setup_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Dual Field Enhanced Phi (Bidirectional)')
    parser.add_argument('--model', type=str, default='microsoft/phi-2',
                      help='Model to use')
    parser.add_argument('--field-sizes', type=int, nargs='+', default=[32],
                      help='Field sizes to use')
    parser.add_argument('--camera', type=int, default=0,  # Changed default to 0 for common systems
                      help='Camera index to use')
    parser.add_argument('--port', type=int, default=7860,
                      help='Port for web interface')
    parser.add_argument('--share', action='store_true',
                      help='Share the interface publicly')
    return parser.parse_args()

def setup_model(args) -> CriticalFieldPhi:
    """Initializes the CriticalFieldPhi model with error handling."""
    try:
        logging.info("Initializing Dual Field Enhanced Phi (Bidirectional)...")
        phi = CriticalFieldPhi(camera_index=args.camera)
        logging.info("Model initialization successful")
        return phi
    except Exception as e:
        logging.error(f"Error initializing model: {str(e)}")
        raise

def main():
    """Main execution with error handling."""
    try:
        # Parse arguments
        args = setup_args()
        
        # Initialize model
        phi = setup_model(args)
        
        # Create interface
        interface = create_interface(phi)
        
        # Launch interface
        interface.launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=args.share
        )
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
