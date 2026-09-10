/*
 * diffraction.js — kinematic (single-scattering) geometry for surface
 * electron diffraction: LEED / RHEED spot-streak positions and RHEED
 * Kikuchi lines.
 *
 * Pure-vanilla-JS port of the verified Python core in
 * src/stm_data_processing/leed_rheed/kinematic.py and kikuchi.py. Only the
 * geometric positions of spots / streaks / lines are computed, never
 * dynamical intensities. Numerics mirror the Python reference operation by
 * operation (IEEE doubles).
 *
 * Loading
 * -------
 * - Browser: <script src="diffraction.js"></script> exposes the global
 *   object `window.Diffraction`.
 * - Node: `require('../app/latticeSIM/ui/diffraction.js')` (CommonJS footer)
 *   returns the same object.
 *
 * Conventions
 * -----------
 * - The surface lies in the xy plane; the outward surface normal points
 *   along +z.
 * - k_in / k_out are the incident / outgoing wave vectors; elastic
 *   scattering fixes |k_in| = |k_out| = k.
 * - The 2D lattice basis vectors a1 = (a1x, a1y) and a2 = (a2x, a2y) lie in
 *   the xy plane.
 * - Lengths are in Angstrom, wave vectors in Angstrom^-1, energies in eV.
 * - Vectors are plain JS arrays of numbers.
 */

(function () {
  'use strict';

  var TWO_PI = 2 * Math.PI;

  /* ---- tiny numeric helpers --------------------------------------- */

  // Return `value` as a 1-D array of exactly `size` finite numbers.
  function asVector(value, size, name) {
    if (
      !value ||
      typeof value !== 'object' ||
      typeof value.length !== 'number' ||
      value.length !== size
    ) {
      throw new Error(name + ' must be a vector of length ' + size);
    }
    var out = new Array(size);
    for (var i = 0; i < size; i++) {
      out[i] = Number(value[i]);
      if (!isFinite(out[i])) {
        throw new Error(name + ' must contain finite numbers');
      }
    }
    return out;
  }

  function dot(u, v) {
    var s = 0;
    for (var i = 0; i < u.length; i++) {
      s += u[i] * v[i];
    }
    return s;
  }

  function norm(v) {
    var s = 0;
    for (var i = 0; i < v.length; i++) {
      s += v[i] * v[i];
    }
    return Math.sqrt(s);
  }

  function scale(v, s) {
    var out = new Array(v.length);
    for (var i = 0; i < v.length; i++) {
      out[i] = v[i] * s;
    }
    return out;
  }

  function sub(u, v) {
    var out = new Array(u.length);
    for (var i = 0; i < u.length; i++) {
      out[i] = u[i] - v[i];
    }
    return out;
  }

  function cross3(u, v) {
    return [
      u[1] * v[2] - u[2] * v[1],
      u[2] * v[0] - u[0] * v[2],
      u[0] * v[1] - u[1] * v[0]
    ];
  }

  /* ---- 2D reciprocal lattice --------------------------------------- */

  // 2D reciprocal basis of a surface lattice: a_i . b_j = 2 pi delta_ij
  // with the scalar 2D cross product cross = a1x*a2y - a1y*a2x,
  //   b1 = 2 pi (a2y, -a2x) / cross,  b2 = 2 pi (-a1y, a1x) / cross.
  function surfaceReciprocal(a1, a2) {
    a1 = asVector(a1, 2, 'a1');
    a2 = asVector(a2, 2, 'a2');
    var cross = a1[0] * a2[1] - a1[1] * a2[0];
    var scaleFactor = Math.hypot(a1[0], a1[1]) * Math.hypot(a2[0], a2[1]);
    if (Math.abs(cross) <= 1e-12 * Math.max(scaleFactor, 1.0)) {
      throw new Error('surface lattice vectors must not be (nearly) collinear');
    }
    var factor = TWO_PI / cross;
    return {
      b1: [a2[1] * factor, -a2[0] * factor],
      b2: [-a1[1] * factor, a1[0] * factor]
    };
  }

  /* ---- 3D reciprocal lattice ---------------------------------------- */

  // 3D reciprocal basis of a lattice with cell volume
  // V = a1 . (a2 x a3):
  //   b1 = 2 pi (a2 x a3) / V, b2 = 2 pi (a3 x a1) / V,
  //   b3 = 2 pi (a1 x a2) / V.
  function reciprocal3d(a1, a2, a3) {
    a1 = asVector(a1, 3, 'a1');
    a2 = asVector(a2, 3, 'a2');
    a3 = asVector(a3, 3, 'a3');
    var cross23 = cross3(a2, a3);
    var volume = dot(a1, cross23);
    var scaleFactor = norm(a1) * norm(a2) * norm(a3);
    if (Math.abs(volume) <= 1e-12 * Math.max(scaleFactor, 1.0)) {
      throw new Error('lattice vectors must span a non-zero 3D volume');
    }
    var factor = TWO_PI / volume;
    return {
      b1: scale(cross23, factor),
      b2: scale(cross3(a3, a1), factor),
      b3: scale(cross3(a1, a2), factor)
    };
  }

  /* ---- free-electron wavenumber ------------------------------------- */

  // de Broglie wavenumber k = 2 pi sqrt(E / 150.4) A^-1; with the
  // relativistic flag the result is multiplied by sqrt(1 + E / 1022000)
  // (2 m_e c^2 = 1022000 eV).
  function electronWavenumber(energy, relativistic) {
    var k = TWO_PI * Math.sqrt(energy / 150.4);
    if (relativistic) {
      k = k * Math.sqrt(1.0 + energy / 1022000.0);
    }
    return k;
  }

  /* ---- Ewald-sphere / rod Laue cuts ---------------------------------- */

  // Cut the Ewald sphere with the reciprocal lattice rods. A rod at
  // g = h*b1 + k*b2 is visible when parallel momentum conservation
  // k_out_par = k_in_par + g leaves a real vertical component
  // k_out_z^2 = k^2 - |k_out_par|^2 >= 0 with k = |k_in|. Rod indices are
  // tested in the rectangle |h| <= h_max, |k| <= k_max; an omitted bound is
  // chosen so that every rod able to intersect the sphere (the region
  // |k_in_par + g| <= k lies inside |g| <= k + |k_in_par|, an ellipse whose
  // axis extents follow from the metric inverse of A^T A, A = [b1 b2]).
  function solveLaue(b1, b2, kIn, hMax, kMax) {
    b1 = asVector(b1, 2, 'b1');
    b2 = asVector(b2, 2, 'b2');
    kIn = asVector(kIn, 3, 'kIn');
    var kk = norm(kIn);
    if (kk === 0.0) {
      throw new Error('k_in must be a non-zero wave vector');
    }
    var kInParX = kIn[0];
    var kInParY = kIn[1];

    if (hMax == null || kMax == null) {
      // Axis extents of the ellipse in (h, k) space:
      // reach * sqrt((M^-1)_ii) with M = A^T A, A = [b1 b2] (columns).
      var m00 = dot(b1, b1);
      var m01 = dot(b1, b2);
      var m11 = dot(b2, b2);
      var det = m00 * m11 - m01 * m01;
      var reach =
        kk + Math.sqrt(kInParX * kInParX + kInParY * kInParY);
      if (hMax == null) {
        hMax = Math.ceil(reach * Math.sqrt(m11 / det));
      }
      if (kMax == null) {
        kMax = Math.ceil(reach * Math.sqrt(m00 / det));
      }
    }
    var hLim = Math.trunc(hMax);
    var kLim = Math.trunc(kMax);

    var tol = 1e-12 * Math.max(kk * kk, 1.0);
    var rods = [];
    for (var h = -hLim; h <= hLim; h++) {
      for (var k = -kLim; k <= kLim; k++) {
        var gx = h * b1[0] + k * b2[0];
        var gy = h * b1[1] + k * b2[1];
        var px = gx + kInParX;
        var py = gy + kInParY;
        var disc = kk * kk - (px * px + py * py);
        if (disc < -tol) {
          continue;
        }
        var kz = Math.sqrt(Math.max(disc, 0.0));
        rods.push({
          h: h,
          k: k,
          g_x: gx,
          g_y: gy,
          // Two exit wave vectors: the +z solution first; the rows
          // coincide at exact tangency of sphere and rod.
          k_out: [[px, py, kz], [px, py, -kz]]
        });
      }
    }
    return rods;
  }

  /* ---- LEED ----------------------------------------------------------- */

  // LEED spot positions for normal incidence: the beam enters along -z,
  // k_in = (0, 0, -k), and only beams leaving the surface (k_out_z > 0) are
  // kept. The screen position of a spot is then simply g.
  function leedPattern(a1, a2, energy, hMax, kMax) {
    var rb = surfaceReciprocal(a1, a2);
    var wavenumber = electronWavenumber(energy);
    var rods = solveLaue(
      rb.b1, rb.b2, [0.0, 0.0, -wavenumber], hMax, kMax
    );
    var spots = [];
    for (var i = 0; i < rods.length; i++) {
      var rod = rods[i];
      for (var j = 0; j < 2; j++) {
        var kOut = rod.k_out[j];
        if (kOut[2] > 0.0) {
          spots.push({
            h: rod.h,
            k: rod.k,
            g_x: rod.g_x,
            g_y: rod.g_y,
            k_out: [kOut[0], kOut[1], kOut[2]]
          });
        }
      }
    }
    return spots;
  }

  /* ---- RHEED ----------------------------------------------------------- */

  // RHEED streak geometry for grazing incidence in the xz plane: the beam
  // travels along +x tilted toward the surface by the grazing angle alpha
  // (angle with the surface), k_in = (k cos(alpha), 0, -k sin(alpha)). Each
  // rod cut by the Ewald sphere yields two exit wave vectors (upward and
  // downward); adjacent rods form the near-grazing streaks.
  function rheedPattern(a1, a2, energy, grazingAngleDeg, hMax, kMax) {
    var rb = surfaceReciprocal(a1, a2);
    var wavenumber = electronWavenumber(energy);
    var alpha = grazingAngleDeg * Math.PI / 180.0;
    var kIn = [
      wavenumber * Math.cos(alpha),
      0.0,
      -wavenumber * Math.sin(alpha)
    ];
    var rods = solveLaue(rb.b1, rb.b2, kIn, hMax, kMax);
    var streaks = [];
    for (var i = 0; i < rods.length; i++) {
      var rod = rods[i];
      for (var j = 0; j < 2; j++) {
        var kOut = rod.k_out[j];
        streaks.push({
          h: rod.h,
          k: rod.k,
          g_x: rod.g_x,
          g_y: rod.g_y,
          k_out: [kOut[0], kOut[1], kOut[2]]
        });
      }
    }
    return streaks;
  }

  /* ---- Kikuchi lines --------------------------------------------------- */

  var INNER_POTENTIAL_COEFF = 0.265; // A^-2 per eV, PyRHEED convention

  // Orthonormal frame (e1, e2) spanning the plane perpendicular to k_in.
  function screenBasis(kIn, k0) {
    var khat = scale(kIn, 1.0 / k0);
    var reference = [0.0, 0.0, 1.0];
    if (Math.abs(dot(khat, reference)) > 0.9) {
      reference = [0.0, 1.0, 0.0];
    }
    var e1 = sub(reference, scale(khat, dot(reference, khat)));
    e1 = scale(e1, 1.0 / norm(e1));
    var e2 = cross3(khat, e1);
    return { e1: e1, e2: e2, khat: khat };
  }

  // Kikuchi-line pairs generated by the 3D reciprocal vectors g. Every
  // g = h*b1 + k*b2 + l*b3 with |h|,|k|,|l| <= index_max is projected onto
  // the screen plane (through the origin, perpendicular to k_in). Vectors
  // (anti-)parallel to k_in are skipped (their pair collapses to a point)
  // and vectors with |g_proj| > k_eff are dropped (outside the
  // refraction-enlarged Ewald sphere). The mean inner potential V0 raises
  // the internal wavenumber to k_eff = sqrt(k0^2 + 0.265*V0).
  function kikuchiLines(b1, b2, b3, kIn, innerPotential, indexMax) {
    b1 = asVector(b1, 3, 'b1');
    b2 = asVector(b2, 3, 'b2');
    b3 = asVector(b3, 3, 'b3');
    kIn = asVector(kIn, 3, 'kIn');
    var k0 = norm(kIn);
    if (k0 === 0.0) {
      throw new Error('k_in must be a non-zero wave vector');
    }
    var pot = (innerPotential == null) ? 0.0 : innerPotential;
    var kEff = Math.sqrt(k0 * k0 + INNER_POTENTIAL_COEFF * pot);
    var iMax = (indexMax == null) ? 5 : Math.trunc(indexMax);
    var frame = screenBasis(kIn, k0);
    var khat = frame.khat;
    var e1 = frame.e1;
    var e2 = frame.e2;

    var zero2 = [0.0, 0.0];
    var zero3 = [0.0, 0.0, 0.0];
    var lines = [];

    function record(type, point2d, point) {
      return {
        h: h,
        k: k,
        l: l,
        g_3d: [g3d[0], g3d[1], g3d[2]],
        g_proj: [gProj[0], gProj[1], gProj[2]],
        normal_2d: [normal2d[0], normal2d[1]],
        direction: [direction[0], direction[1], direction[2]],
        type: type,
        point_2d: [point2d[0], point2d[1]],
        point: [point[0], point[1], point[2]]
      };
    }

    for (var h = -iMax; h <= iMax; h++) {
      for (var k = -iMax; k <= iMax; k++) {
        for (var l = -iMax; l <= iMax; l++) {
          if (h === 0 && k === 0 && l === 0) {
            continue;
          }
          var g3d = [
            h * b1[0] + k * b2[0] + l * b3[0],
            h * b1[1] + k * b2[1] + l * b3[1],
            h * b1[2] + k * b2[2] + l * b3[2]
          ];
          var gNorm = norm(g3d);
          var gAlong = dot(g3d, khat);
          var gProj = [
            g3d[0] - gAlong * khat[0],
            g3d[1] - gAlong * khat[1],
            g3d[2] - gAlong * khat[2]
          ];
          var pNorm = norm(gProj);
          if (pNorm <= 1e-9 * gNorm) {
            continue; // g (anti-)parallel to k_in: pair collapses
          }
          if (pNorm > kEff * (1.0 + 1e-12)) {
            continue; // outside the refracted Ewald sphere
          }
          var p2d = [dot(gProj, e1), dot(gProj, e2)];
          var n2dLen = norm(p2d);
          var normal2d = [p2d[0] / n2dLen, p2d[1] / n2dLen];
          var unit = [gProj[0] / pNorm, gProj[1] / pNorm, gProj[2] / pNorm];
          var direction = cross3(khat, unit);
          lines.push(record('excess', p2d, gProj));
          lines.push(record('deficient', zero2, zero3));
        }
      }
    }

    return {
      k0: k0,
      k_eff: kEff,
      innerPotential: pot,
      indexMax: iMax,
      lines: lines
    };
  }

  /* ---- exports ---------------------------------------------------------- */

  var Diffraction = {
    surfaceReciprocal: surfaceReciprocal,
    reciprocal3d: reciprocal3d,
    electronWavenumber: electronWavenumber,
    solveLaue: solveLaue,
    leedPattern: leedPattern,
    rheedPattern: rheedPattern,
    kikuchiLines: kikuchiLines
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = Diffraction;
  }
  if (typeof window !== 'undefined') {
    window.Diffraction = Diffraction;
  }
})();
