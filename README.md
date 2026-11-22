# Physics-Informed Neural Networks (PINNs) for 2D Axisymmetric Taylor Dispersion

A mesh-free **PyTorch Physics-Informed Neural Network (PINN)** solver for transient **2D axisymmetric Taylor-Aris dispersion** in laminar pipe flow. This project models how a finite solute slug shears under parabolic Hagen-Poiseuille flow and relaxes via radial molecular diffusion into the classical 1D Gaussian dispersion regime—validated directly against the analytical **Aris-Taylor asymptotic limit** and global solute mass conservation.

![Taylor Dispersion Animation](taylor_dispersion_graetz_theme.gif)
*(Replace filename above with your exported `.gif` or `.mp4` preview path)*

---

## Key Features & Numerical Methodology

* **Co-Moving Reference Frame (\(\xi = z^* - t^*\)):** Eliminates long-range bulk advection bottlenecks by transforming into a Galilean frame translating at the cross-sectional mean velocity (\(\bar{u}^* = 1\)), isolating relative shear-induced spreading (\(u_{\text{rel}}(r^*) = 1 - 2{r^*}^2\)) inside a compact computational domain.
* **Cylindrical Axis Singularity Handling:** Resolves the \(\frac{1}{r^*}\frac{\partial c}{\partial r^*}\) coordinate singularity along the tube centerline (\(r^* = 0\)) using L'Hôpital's limit (\(\frac{1}{r^*}\frac{\partial c}{\partial r^*} \to \frac{\partial^2 c}{\partial {r^*}^2}\) as \(r^* \to 0\)).
* **Hard-Enforced Physical Positivity:** Applies a `Softplus` output activation (β = 5.0) to guarantee strictly non-negative solute concentrations (c ≥ 0) and prevent Gibbs undershoot ripples near sharp concentration fronts.
* **Three-Stage Optimization Curriculum:**
  1. **Causal Adam Phase:** Enforces temporal causality via chronological residual chunk weighting (\(w_i = \exp(-\epsilon \sum_{k < i} \mathcal{L}_k)\)) paired with periodic interior collocation resampling.
  2. **Unweighted Adam Warm-Up:** Smoothly transitions the loss landscape from causal weighting to standard unweighted MSE.
  3. **L-BFGS Fine-Tuning:** Polishes the solution using second-order curvature (`strong_wolfe` line search) to achieve **< 0.05% total solute mass error**.
* **Asymptotic Analytical Validation:** Computes the cross-sectional area-averaged concentration \(\bar{c}(\xi, t^*) = 2 \int_0^1 c(r^*, \xi, t^*)\,r^*\,dr^*\) and benchmarks it against the exact 1D Aris-Taylor error-function solution.

---

## Mathematical Formulation

### 1. Nondimensionalization & Galilean Transformation
We consider transient solute transport in a cylindrical tube of radius R under fully developed laminar Poiseuille flow with cross-sectional mean velocity \(U_{\text{avg}}\) and molecular diffusivity D. Defining dimensionless coordinates using the tube radius R and advective timescale \(R / U_{\text{avg}}\):

\[r^* = \frac{r}{R}, \qquad z^* = \frac{z}{R}, \qquad t^* = \frac{U_{\text{avg}}\,t}{R}, \qquad \text{Pe} = \frac{U_{\text{avg}} R}{D}\]

Applying the co-moving coordinate transformation \(\xi = z^* - t^*\) yields the governing dimensionless PDE:

\[\frac{\partial c}{\partial t^*} + (1 - 2{r^*}^2)\frac{\partial c}{\partial \xi} = \frac{1}{\text{Pe}}\left(\frac{\partial^2 c}{\partial \xi^2} + \frac{\partial^2 c}{\partial {r^*}^2} + \frac{1}{r^*}\frac{\partial c}{\partial r^*}\right)\]

### 2. Initial & Boundary Conditions
* **Initial Top-Hat Slug (\(t^* = 0\)):** Smooth differentiable pulse of concentration C₀ = 1 centered at \(\xi \in [-0.5, 0.5]\).
* **Impermeable Tube Wall (\(r^* = 1\)):** Zero radial mass flux, \(\frac{\partial c}{\partial r^*} = 0\).
* **Centerline Symmetry (\(r^* = 0\)):** Smooth radial profile, \(\frac{\partial c}{\partial r^*} = 0\).
* **Far-Field Axial Boundaries (\(\xi = \xi_{\min}, \xi_{\max}\)):** Zero solute concentration, c = 0.

### 3. Aris-Taylor Asymptotic Limit
For \(t^* \gtrsim 0.25\,\text{Pe}\), radial diffusion homogenizes cross-sectional concentration gradients, and the area-averaged concentration \(\bar{c}(\xi, t^*)\) converges to the 1D effective diffusion equation (Taylor, *Proc. R. Soc. A* 1953; Aris, *Proc. R. Soc. A* 1956):

\[\frac{\partial \bar{c}}{\partial t^*} = D_{\text{eff}}\frac{\partial^2 \bar{c}}{\partial \xi^2}, \qquad D_{\text{eff}} = \frac{1}{\text{Pe}} + \frac{\text{Pe}}{48}\]

---

## Repository Structure

```text
├── taylor_dispersion_pinn.ipynb        # Main PyTorch PINN training, validation & visualization notebook
├── taylor_dispersion_graetz_theme.mp4  # Dual-panel 2D contour & 1D Aris-Taylor animation
└── README.md                           # Project documentation and mathematical formulation
```

---

## Quick Start

### Prerequisites
* Python 3.9+
* PyTorch (CUDA recommended)
* NumPy, SciPy, Matplotlib
* `ffmpeg` (for `.mp4` rendering) or `pillow` (for `.gif` fallback)

### Installation & Execution
```bash
git clone https://github.com<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install torch numpy scipy matplotlib pillow
```
Open `taylor_dispersion_pinn.ipynb` and run all cells sequentially to train the network, verify mass conservation, and export the dual-panel animation.

---

## References
1. **G. I. Taylor**, *"Dispersion of soluble matter in solvent flowing slowly through a tube,"* Proceedings of the Royal Society A, **219**, 186–203 (1953).
2. **R. Aris**, *"On the dispersion of a solute in a fluid flowing through a tube,"* Proceedings of the Royal Society A, **235**, 67–77 (1956).
3. **S. Wang, S. Sankaran, & P. Perdikaris**, *"Respecting causality is all you need for training physics-informed neural networks,"* Computer Methods in Applied Mechanics and Engineering, **421**, 116813 (2024).
