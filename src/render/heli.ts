// The CH-47J as the renderer and the client see it: where the moving parts
// sit on the airframe. Mirrors scripts/build-ch47j.py (it prints these).
// Model frame: +Z nose, +Y up, +X left; origin on the ground under the middle
// of the fuselage, the wheels touching y = 0 with the struts compressed.

import type { V3 } from './math';

/** Floats per helicopter handed to `Renderer.setHelis`:
 * x, y, z, quaternion (x, y, z, w), tint (1 = intact .. 0 = burnt out),
 * rotorAngle (rad), gear (0 = struts compressed on the ground .. 1 = extended in flight .. 2 = stowed),
 * rotor speed (optional: 0 = worked out from how fast rotorAngle changes, else 0..1 of full speed;
 * it blurs the blades into a disc and lifts them from their droop). */
export const HELI_STRIDE = 11;

/** Rotor radius (m), three blades each. */
export const HELI_ROTOR_RADIUS = 9.15;
/** Full rotor speed (rad/s): 225 rpm. */
export const HELI_ROTOR_SPEED = 23.6;
/** Hub centres; `tilt` leans the shaft forward (rad). The forward rotor turns
 * anticlockwise seen from above by `rotorAngle`, the aft one (a mirror image) the other way, 60 degrees out of phase. */
export const HELI_ROTORS: { pos: V3; tilt: number; dir: 1 | -1; phase: number }[] = [
  { pos: [0, 4.5, 5.75], tilt: 0.0873, dir: 1, phase: 0 },
  { pos: [0, 5.6, -6.15], tilt: 0.0436, dir: -1, phase: Math.PI / 3 },
];

/** The blades are modelled straight and bent in the vertex shader: nothing
 * inside BLADE_ROOT, then a parabola to the tip, which hangs at BLADE_REST (m)
 * with the rotor stopped and cones up to BLADE_CONE at full speed. */
export const HELI_BLADE_ROOT = 0.6;
export const HELI_BLADE_REST = -1.3;
export const HELI_BLADE_CONE = 0.25;

/** Landing gear: where each sliding strut meets the pod, struts compressed.
 * `gear` 0..1 lets the oleo struts extend downwards by HELI_GEAR_TRAVEL when
 * the weight comes off; 1..2 slides the unit up by `retract` (from the
 * compressed position) into the well under the pod. The real aircraft's gear
 * is fixed: the retraction is a game addition. */
export const HELI_GEAR: { pos: V3; rear: boolean; retract: number }[] = [
  { pos: [1.4, 0.77, 2.4], rear: false, retract: 0.65 },
  { pos: [-1.4, 0.77, 2.4], rear: false, retract: 0.65 },
  { pos: [1.3, 1.033, -4.45], rear: true, retract: 0.913 },
  { pos: [-1.3, 1.033, -4.45], rear: true, retract: 0.913 },
];
export const HELI_GEAR_TRAVEL = 0.2;
export const HELI_WHEEL_RADIUS = 0.32;

/** Mouths of the rocket tubes (four on each side pod); missiles leave along +Z. */
export const HELI_MISSILE_PORTS: V3[] = [
  [2.055, 1.185, 2.45], [2.055, 1.395, 2.45], [2.265, 1.185, 2.45], [2.265, 1.395, 2.45],
  [-2.055, 1.185, 2.45], [-2.055, 1.395, 2.45], [-2.265, 1.185, 2.45], [-2.265, 1.395, 2.45],
];

/** The hull as a box (cabin and side pods): centre and half extents. */
export const HELI_HULL_CENTER: V3 = [0, 1.8, 0];
export const HELI_HULL_HALF: V3 = [1.87, 1.25, 7.8];
/** Top of the aft rotor head above the ground. */
export const HELI_HEIGHT = 5.8;
/** Where a pilot's eyes are (behind the left windscreen panel). */
export const HELI_COCKPIT: V3 = [0.6, 2.35, 6.0];
