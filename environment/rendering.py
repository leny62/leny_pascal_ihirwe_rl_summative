"""PyOpenGL renderer for the block.

Imported lazily from UmurimaEnv.render so training never opens a GL context.

macOS caps at OpenGL 4.1 core: requesting a 3.3 core profile yields 4.1 with
GLSL 4.10 over Metal. Shaders target #version 330 core. No compute shaders.
"""

from __future__ import annotations

from typing import Any

import numpy as np

WINDOW_SIZE = (1280, 720)
GL_MAJOR, GL_MINOR = 3, 3

# Terrace surface, dry ochre to dark wet brown, indexed by depletion fraction
SOIL_DRY_RGB = (0.72, 0.58, 0.36)
SOIL_WET_RGB = (0.28, 0.20, 0.13)
# Crop, healthy green to chlorotic yellow, indexed by nitrogen stress
CROP_HEALTHY_RGB = (0.22, 0.55, 0.18)
CROP_STRESSED_RGB = (0.78, 0.74, 0.26)

VERTEX_SHADER = """#version 330 core
layout (location = 0) in vec3 in_pos;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec3 in_offset;
layout (location = 3) in vec3 in_colour;
layout (location = 4) in float in_scale;

uniform mat4 u_view_proj;

out vec3 v_normal;
out vec3 v_colour;

void main() {
    vec3 world = in_pos * vec3(1.0, in_scale, 1.0) + in_offset;
    gl_Position = u_view_proj * vec4(world, 1.0);
    v_normal = in_normal;
    v_colour = in_colour;
}
"""

FRAGMENT_SHADER = """#version 330 core
in vec3 v_normal;
in vec3 v_colour;

uniform vec3 u_light_dir;
uniform vec3 u_light_colour;

out vec4 frag_colour;

void main() {
    float lambert = max(dot(normalize(v_normal), normalize(-u_light_dir)), 0.0);
    vec3 lit = v_colour * (0.35 + 0.65 * lambert) * u_light_colour;
    frag_colour = vec4(lit, 1.0);
}
"""


class BlockRenderer:
    """Draws the four terrace zones, crop, weather and HUD."""

    def __init__(self, render_mode: str, size: tuple[int, int] = WINDOW_SIZE) -> None:
        self.render_mode = render_mode
        self.size = size
        self._closed = False
        # TODO: init pygame, request a 3.3 core context (hidden when rgb_array),
        # compile shaders, build the terrain VAO and the instance buffers
        raise NotImplementedError

    def draw(self, state: dict[str, Any]) -> np.ndarray | None:
        """Render one frame. Returns an (H, W, 3) uint8 array in rgb_array mode."""
        # TODO: update instance buffers from state, draw terrain, crop, weeds,
        # crew, rain, then the HUD quad. glReadPixels and flip rows for rgb_array.
        raise NotImplementedError

    def _build_hud(self, state: dict[str, Any]) -> np.ndarray:
        """Text overlay. Pre-harvest interval line goes red inside the window."""
        raise NotImplementedError

    def close(self) -> None:
        """Idempotent, called more than once in practice."""
        if self._closed:
            return
        self._closed = True
        # TODO: delete GL objects, pygame.quit()
