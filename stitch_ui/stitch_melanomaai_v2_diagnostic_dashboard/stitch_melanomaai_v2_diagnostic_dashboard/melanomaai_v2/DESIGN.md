---
name: MelanomaAI v2
colors:
  surface: '#0f141b'
  surface-dim: '#0f141b'
  surface-bright: '#343942'
  surface-container-lowest: '#090e16'
  surface-container-low: '#171c24'
  surface-container: '#1b2028'
  surface-container-high: '#252a33'
  surface-container-highest: '#30353e'
  on-surface: '#dee2ee'
  on-surface-variant: '#c2c6d6'
  inverse-surface: '#dee2ee'
  inverse-on-surface: '#2c3139'
  outline: '#8c909f'
  outline-variant: '#424754'
  surface-tint: '#adc6ff'
  primary: '#adc6ff'
  on-primary: '#002e6a'
  primary-container: '#4d8eff'
  on-primary-container: '#00285d'
  inverse-primary: '#005ac2'
  secondary: '#4cd7f6'
  on-secondary: '#003640'
  secondary-container: '#03b5d3'
  on-secondary-container: '#00424e'
  tertiary: '#c3c6cf'
  on-tertiary: '#2d3137'
  tertiary-container: '#8d9199'
  on-tertiary-container: '#262a31'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#d8e2ff'
  primary-fixed-dim: '#adc6ff'
  on-primary-fixed: '#001a42'
  on-primary-fixed-variant: '#004395'
  secondary-fixed: '#acedff'
  secondary-fixed-dim: '#4cd7f6'
  on-secondary-fixed: '#001f26'
  on-secondary-fixed-variant: '#004e5c'
  tertiary-fixed: '#dfe2eb'
  tertiary-fixed-dim: '#c3c6cf'
  on-tertiary-fixed: '#181c22'
  on-tertiary-fixed-variant: '#43474e'
  background: '#0f141b'
  on-background: '#dee2ee'
  surface-variant: '#30353e'
typography:
  display-lg:
    fontFamily: Geist
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.04em
  headline-lg:
    fontFamily: Geist
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Geist
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.02em
  body-lg:
    fontFamily: Geist
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
    letterSpacing: -0.01em
  body-md:
    fontFamily: Geist
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
    letterSpacing: -0.01em
  label-md:
    fontFamily: Geist
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
    letterSpacing: 0.02em
  label-sm:
    fontFamily: Geist
    fontSize: 12px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.05em
  headline-lg-mobile:
    fontFamily: Geist
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 36px
    letterSpacing: -0.02em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  unit: 8px
  container-padding-desktop: 40px
  container-padding-mobile: 20px
  gutter: 24px
  stack-sm: 12px
  stack-md: 24px
  stack-lg: 48px
---

## Brand & Style

This design system is engineered for high-stakes medical precision and futuristic performance. It targets healthcare professionals and researchers who require an environment that feels both clinical and cutting-edge. 

The aesthetic is **Modern Glassmorphism** fused with **Technical Minimalism**. The interface should feel like a sophisticated cockpit, utilizing deep canvases, translucent overlays, and luminous data points to reduce cognitive load while maintaining an aura of advanced artificial intelligence. Every interaction must reinforce feelings of trust, accuracy, and ultra-fast processing.

## Colors

The palette is anchored in a "Deep Space" hierarchy to ensure maximum contrast for medical imagery. 

- **Primary Canvas (#0A0E14):** The deepest base layer.
- **Surface Layer (#12171F):** Used for primary UI containers and cards.
- **Accents:** Electric Blue and Medical Cyan are used for interactive elements and AI-assisted highlights.
- **Semantic Feedback:** High-saturation Red and Green are reserved strictly for diagnosis results, ensuring they command immediate attention without overwhelming the rest of the UI.

## Typography

The design system utilizes **Geist** for its monospaced-influenced precision and technical clarity. 

- **Tracking:** Headings use negative letter-spacing to create a "locked-in," premium feel similar to high-end SaaS tools.
- **Hierarchy:** Use `label-sm` for technical metadata and secondary information to maintain a clean, organized look.
- **Contrast:** Utilize varying font weights (from 400 to 700) rather than color shifts alone to distinguish information types on dark backgrounds.

## Layout & Spacing

The layout follows a **Fixed-Fluid Hybrid** model. Navigation and sidebars are fixed, while main diagnostic panels use a fluid 12-column grid.

- **Base Unit:** An 8px rhythm governs all padding and margins.
- **Breathability:** Given the intensity of dark mode and medical data, generous whitespace (48px+) is used between major sections to prevent visual fatigue.
- **Micro-alignment:** Interactive elements within cards should use a 12px `stack-sm` to maintain a dense, high-performance data density.

## Elevation & Depth

Depth is achieved through physical layering and light simulation rather than traditional drop shadows.

- **The Glass Layer:** Primary containers use a `backdrop-filter: blur(20px)` with a background opacity of 60% of the Surface color (#12171F).
- **Micro-borders:** All containers must feature a 1px solid border. Use `rgba(255, 255, 255, 0.08)` for standard borders and `rgba(59, 130, 246, 0.3)` for active/focused states.
- **Inner Glows:** Interactive components (cards, buttons) should have a subtle top-down inner shadow (`inset 0 1px 0 rgba(255, 255, 255, 0.1)`) to simulate a light source from above.
- **Zero Shadows:** Avoid heavy black shadows; the contrast between the dark navy background and the blurred charcoal surfaces provides natural separation.

## Shapes

The design system employs a sophisticated "soft-tech" corner radius.

- **Base Radius:** Standard UI elements like buttons and inputs use a 0.5rem (8px) radius.
- **Large Components:** Diagnostic cards and main containers must use `rounded-xl` (1.5rem / 24px) to emphasize the "Glassmorphic" sheet metaphor.
- **Contextual Rounding:** Ensure that nested elements have a smaller radius than their parents to maintain visual harmony (Radius Parent - Padding = Radius Child).

## Components

### Buttons & Inputs
- **Primary Action:** Solid Electric Blue with a subtle cyan outer glow on hover. No text shadows.
- **Ghost Action:** 1px micro-border with high-blur backdrop.
- **Inputs:** Darker than the surface color, with a 1px border that illuminates into Medical Cyan on focus.

### Cards (Diagnostic)
- These are the core of the system. They feature the 24px radius, 20px backdrop blur, and a 1px micro-border. 
- Header areas within cards should be separated by a subtle 1px horizontal line (`rgba(255,255,255,0.05)`).

### Status Indicators
- **Glow Dots:** For AI status (e.g., "Analyzing"), use a pulsing 8px circle with a 12px Gaussian blur of the same color.
- **Diagnosis Chips:** Use high-contrast backgrounds for Malignant/Benign results with white text to ensure immediate readability.

### Medical Iconography
- Icons should be linear, 1.5px stroke width, using the Medical Cyan color for "active" states and a muted grey-blue for "inactive" states. Avoid filled icons to keep the UI feeling "light" despite the dark theme.