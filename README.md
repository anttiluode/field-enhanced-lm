# Field-Enhanced Language Model (FELM)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An experimental implementation of a field-enhanced language model that integrates electromagnetic field dynamics with transformer-based language models. Inspired by the Conscious Electromagnetic Information (CEMI) field theory.

## 🌟 Features

- Integration of physical field dynamics with language model processing
- Real-time visualization of field states and patterns
- Bidirectional coupling between visual and language fields
- Interactive parameter tuning for field dynamics
- Pattern detection and criticality analysis
- Streaming token generation with field influence

## 🔍 Overview

This project explores the integration of simulated electromagnetic fields with language models, inspired by theories about the role of EM fields in biological consciousness. The system consists of:

- A base language model (Phi-2)
- Two coupled critical fields (visual and language)
- Real-time pattern detection and analysis
- Field-influenced token generation
- Interactive visualization interface

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/anttiluode/field-enhanced-lm.git
cd field-enhanced-lm

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Unix
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers
- gradio
- plotly
- numpy
- scipy
- powerlaw
- opencv-python

## 🚀 Usage

Run the main interface:

```bash
python main.py --model microsoft/phi-2 --field-sizes 32

or simply

python main.py --model microsoft/phi-2

you can also have camera input in the visual feed:

python main.py --camera 0  # Change the index number if your camera is in another index

Without the camera the visual field is just another field that tries to resonate with field 1. 

```

The language model may talk indefinetly as the field interactions may break token limits. 

### Parameters

#### Language Field Parameters:
- `coupling_strength` (0.0-1.0): Field-model interaction strength
- `damping` (0.0-1.0): Field activity decay rate
- `tension` (0.0-5.0): Pattern propagation speed
- `nonlinearity` (0.0-1.0): Field response curve steepness

#### Visual Field Parameters:
Similar parameters for visual field dynamics

#### Cross-Modal Parameters:
- `cross_coupling_strength` (0.0-1.0): Strength of field interactions

## 🔬 Technical Details

### Field Dynamics

The field evolution follows a modified complex Ginzburg-Landau equation:

```python
d_magnitude = (
    self.tension * magnitude_lap -
    self.damping * self.magnitude -
    self.nonlinearity * self.magnitude**3
)
```

### Pattern Detection

Patterns are detected using adaptive thresholding and connected component analysis, with metrics for:
- Power law exponents
- Correlation lengths
- Fractal dimensions
- Field complexity

## 📊 Visualization

The interface provides real-time visualization of:
- Field magnitudes and phases
- Pattern detection results
- Prediction errors and coupling strengths
- Field metrics and stability measures

## 🧪 Experimental Results

Current observations include:
- Emergence of coherent field patterns during processing
- Variable processing speeds and "thinking" pauses
- Token limit breaking through field-mediated state persistence
- Self-organizing criticality in field dynamics

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) first.

Areas of particular interest:
- Field dynamics optimization
- Pattern detection improvements
- Alternative coupling mechanisms
- Theoretical framework development

## 📚 References

1. McFadden, J. (2020). Integrating information in the brain's EM field: the cemi field theory of consciousness
2. Freeman, W. J. (2007). Scale-free neocortical dynamics
3. Transformer architecture papers

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This is experimental research software. The field-enhancement mechanisms are highly experimental and based on theoretical proposals about consciousness and information processing.

## 🙏 Acknowledgments

- Microsoft for the Phi-2 model
- Theoretical insights from CEMI field theory
- Contributors and feedback from the research community
