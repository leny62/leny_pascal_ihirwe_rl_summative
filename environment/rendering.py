"""PyOpenGL renderer for the block.

Imported lazily from UmurimaEnv.render so training never opens a GL context.

macOS caps at OpenGL 4.1 core with GLSL 4.10 over Metal. Shaders target
#version 410 core. The core profile must be requested explicitly via pygame GL
attributes before creating the window, otherwise the default legacy 2.1 context
is returned. No compute shaders.

In human mode the view is orbitable: drag to rotate, scroll to zoom, R to reset.
Animation is driven off wall-clock time rather than the simulation day, so rain
falls smoothly regardless of how fast the policy is stepped.
"""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np

WINDOW_SIZE = (1280, 720)
GL_MAJOR, GL_MINOR = 3, 3

SOIL_DRY_RGB = (0.72, 0.58, 0.36)
SOIL_WET_RGB = (0.34, 0.24, 0.15)
SUBSOIL_RGB = (0.30, 0.22, 0.16)
CROP_HEALTHY_RGB = (0.22, 0.55, 0.18)
CROP_STRESSED_RGB = (0.78, 0.74, 0.26)
RAIN_RGB = (0.72, 0.85, 0.98)

SKY_ZENITH_RGB = (0.24, 0.45, 0.72)
SKY_HORIZON_RGB = (0.68, 0.80, 0.90)
SKY_ZENITH_WET_RGB = (0.34, 0.40, 0.48)
SKY_HORIZON_WET_RGB = (0.62, 0.66, 0.70)

# Terrace geometry. Four benches descending from ridge (zone 0) to valley
# bottom (zone 3), matching ZONE_DEPTH_MEAN_M in custom_env.
TERRAIN_COLS = 48
TERRAIN_ROWS = 24
ROWS_PER_ZONE = TERRAIN_ROWS // 4
ZONE_HEIGHTS = (1.35, 0.95, 0.58, 0.24)
TOTAL_WIDTH = 7.0
TOTAL_DEPTH = 4.0
ZONE_DEPTH = TOTAL_DEPTH / 4.0
BASE_Y = -0.30
# Terraces are built with a raised lip on the downslope edge to hold irrigation
# water on the bench instead of letting it run straight off.
BUND_HEIGHT = 0.07
BUND_ROWS = frozenset({ROWS_PER_ZONE - 1, 2 * ROWS_PER_ZONE - 1, 3 * ROWS_PER_ZONE - 1, TERRAIN_ROWS})

# Crop is planted in rows, which is how the block is actually managed and reads
# far better than a random scatter.
CROP_ROWS_PER_ZONE = 3
CROP_PER_ROW = 16
CROP_PER_ZONE = CROP_ROWS_PER_ZONE * CROP_PER_ROW
CROP_RADIUS = 0.045
CROP_MAX_HEIGHT = 0.55

RAIN_INSTANCES = 220
RAIN_TOP = ZONE_HEIGHTS[0] + 2.4
RAIN_BOTTOM = BASE_Y
RAIN_SPAN = RAIN_TOP - RAIN_BOTTOM
RAIN_SPEED = 4.2  # world units per second

HUD_SIZE = (984, 210)
# Per-zone panel on the right of the HUD. Without it the four benches are only
# distinguishable by soil shade, which is hard to read on a projector.
ZONE_PANEL_X = 578
ZONE_PANEL_BARS = (
    ("WATER", (96, 168, 232)),
    ("CANOPY", (104, 184, 108)),
    ("PEST", (226, 112, 104)),
    ("WEED", (226, 182, 96)),
)
ZONE_BAR_W = 62
ZONE_BAR_H = 9
ZONE_BAR_GAP = 15
ZONE_ROW_H = 30
ZONE_LABELS = ("Z0 ridge", "Z1", "Z2", "Z3 valley")
HUD_MARGIN = 14

HUD_TEXT_RGB = (208, 216, 224)
HUD_BRIGHT_RGB = (238, 242, 246)
HUD_AMBER_RGB = (250, 196, 96)
HUD_RED_RGB = (244, 132, 120)
HUD_GREEN_RGB = (168, 208, 168)
HUD_BLUE_RGB = (128, 190, 240)
HUD_DIM_RGB = (120, 134, 148)

# Orbit camera limits
CAMERA_TARGET = (3.4, 0.30, 2.0)
CAMERA_START = (136.2, 33.1, 11.9)  # azimuth, elevation (degrees), distance
ELEVATION_RANGE = (9.0, 78.0)
DISTANCE_RANGE = (5.5, 28.0)
ORBIT_SENSITIVITY = 0.32
ZOOM_SENSITIVITY = 0.08

MESH_VERTEX_SHADER = """#version 410 core
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

MESH_FRAGMENT_SHADER = """#version 410 core
in vec3 v_normal;
in vec3 v_colour;

uniform vec3 u_light_dir;
uniform vec3 u_light_colour;
uniform float u_unlit;

out vec4 frag_colour;

void main() {
    float lambert = max(dot(normalize(v_normal), normalize(-u_light_dir)), 0.0);
    vec3 lit = v_colour * (0.42 + 0.58 * lambert) * u_light_colour;
    frag_colour = vec4(mix(lit, v_colour, u_unlit), 1.0);
}
"""

SCREEN_VERTEX_SHADER = """#version 410 core
layout (location = 0) in vec2 in_pos;
layout (location = 1) in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
}
"""

SKY_FRAGMENT_SHADER = """#version 410 core
in vec2 v_uv;
uniform vec3 u_zenith;
uniform vec3 u_horizon;
out vec4 frag_colour;
void main() {
    frag_colour = vec4(mix(u_horizon, u_zenith, pow(v_uv.y, 0.7)), 1.0);
}
"""

HUD_FRAGMENT_SHADER = """#version 410 core
in vec2 v_uv;
uniform sampler2D u_texture;
out vec4 frag_colour;
void main() {
    frag_colour = texture(u_texture, v_uv);
}
"""


# ---------------------------------------------------------------------------
# shader helpers
# ---------------------------------------------------------------------------


def _compile_shader(src: str, shader_type: int):
    from OpenGL.GL import glCompileShader, glCreateShader, glGetShaderInfoLog, glShaderSource

    shader = glCreateShader(shader_type)
    glShaderSource(shader, src)
    glCompileShader(shader)
    log = glGetShaderInfoLog(shader)
    if log and log.strip():
        raise RuntimeError(f"shader compile error: {log.decode()}")
    return shader


def _link_program(vertex_src: str, fragment_src: str):
    from OpenGL.GL import (
        GL_FRAGMENT_SHADER,
        GL_VERTEX_SHADER,
        glAttachShader,
        glCreateProgram,
        glDeleteShader,
        glGetProgramInfoLog,
        glLinkProgram,
    )

    vs = _compile_shader(vertex_src, GL_VERTEX_SHADER)
    fs = _compile_shader(fragment_src, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)
    glDeleteShader(vs)
    glDeleteShader(fs)
    log = glGetProgramInfoLog(program)
    if log and log.strip():
        raise RuntimeError(f"link error: {log.decode()}")
    return program


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def _vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals, then normalised."""
    normals = np.zeros_like(vertices)
    for tri in triangles:
        a, b, c = vertices[tri]
        normals[tri] += np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    return (normals / lengths).astype(np.float32)


def _zone_of_row(row: int) -> int:
    return min(row // ROWS_PER_ZONE, 3)


def _surface_height(row: int) -> float:
    """Bench height at a grid row, including the retaining bund.

    Shared by the surface and the skirt so the two always meet.
    """
    height = ZONE_HEIGHTS[_zone_of_row(row)]
    return height + BUND_HEIGHT if row in BUND_ROWS else height


def _terrace_surface() -> tuple[np.ndarray, np.ndarray]:
    positions = np.zeros(((TERRAIN_ROWS + 1) * (TERRAIN_COLS + 1), 3), dtype=np.float32)
    i = 0
    for row in range(TERRAIN_ROWS + 1):
        z = (row / TERRAIN_ROWS) * TOTAL_DEPTH
        y = _surface_height(row)
        for col in range(TERRAIN_COLS + 1):
            positions[i] = ((col / TERRAIN_COLS) * TOTAL_WIDTH, y, z)
            i += 1

    indices: list[int] = []
    for row in range(TERRAIN_ROWS):
        for col in range(TERRAIN_COLS):
            tl = row * (TERRAIN_COLS + 1) + col
            tr, bl = tl + 1, tl + TERRAIN_COLS + 1
            br = bl + 1
            indices.extend((tl, bl, tr, tr, bl, br))
    return positions, np.asarray(indices, dtype=np.uint32)


def _terrace_skirt() -> list[tuple[float, float, float]]:
    """Side and bottom faces extruding the benches down to BASE_Y.

    Wound counter-clockwise as seen from outside the block, because face
    culling stays enabled for the scene pass.
    """
    width, depth = TOTAL_WIDTH, TOTAL_DEPTH
    tris: list[tuple[float, float, float]] = []

    for row in range(TERRAIN_ROWS):
        z0 = (row / TERRAIN_ROWS) * depth
        z1 = ((row + 1) / TERRAIN_ROWS) * depth
        y0, y1 = _surface_height(row), _surface_height(row + 1)

        # Left wall, outward normal -x.
        tris += [
            (0.0, y0, z0), (0.0, BASE_Y, z0), (0.0, BASE_Y, z1),
            (0.0, y0, z0), (0.0, BASE_Y, z1), (0.0, y1, z1),
        ]
        # Right wall, outward normal +x.
        tris += [
            (width, y0, z0), (width, BASE_Y, z1), (width, BASE_Y, z0),
            (width, y0, z0), (width, y1, z1), (width, BASE_Y, z1),
        ]

    front_y, back_y = _surface_height(0), _surface_height(TERRAIN_ROWS)
    # Ridge face, outward normal -z.
    tris += [
        (0.0, front_y, 0.0), (width, BASE_Y, 0.0), (0.0, BASE_Y, 0.0),
        (0.0, front_y, 0.0), (width, front_y, 0.0), (width, BASE_Y, 0.0),
    ]
    # Valley face, outward normal +z.
    tris += [
        (0.0, back_y, depth), (0.0, BASE_Y, depth), (width, BASE_Y, depth),
        (0.0, back_y, depth), (width, BASE_Y, depth), (width, back_y, depth),
    ]
    # Underside, outward normal -y.
    tris += [
        (0.0, BASE_Y, 0.0), (width, BASE_Y, 0.0), (width, BASE_Y, depth),
        (0.0, BASE_Y, 0.0), (width, BASE_Y, depth), (0.0, BASE_Y, depth),
    ]
    return tris


def _build_terrain():
    """Terrace mesh with earth base.

    Returns (vao, index_count, colour_vbo, tint, n_surface_vertices).
    """
    import OpenGL.GL as gl

    surface, surface_indices = _terrace_surface()
    surface_normals = _vertex_normals(surface, surface_indices.reshape(-1, 3))
    n_surface = len(surface)

    skirt = np.asarray(_terrace_skirt(), dtype=np.float32)
    skirt_indices = np.arange(len(skirt), dtype=np.uint32).reshape(-1, 3)
    skirt_normals = _vertex_normals(skirt, skirt_indices)

    positions = np.vstack([surface, skirt])
    normals = np.vstack([surface_normals, skirt_normals])
    index_array = np.concatenate([surface_indices, skirt_indices.ravel() + n_surface])

    # Static per-vertex tint so flat soil reads as soil rather than as a plane.
    tint = np.random.default_rng(7).uniform(0.90, 1.10, size=len(positions)).astype(np.float32)

    vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vao)
    _static_attrib(0, positions)
    _static_attrib(1, normals)

    colour_vbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, colour_vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, positions.nbytes, None, gl.GL_DYNAMIC_DRAW)
    gl.glVertexAttribPointer(3, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
    gl.glEnableVertexAttribArray(3)

    # The terrain is drawn as a single instance, so offset and scale are constant.
    _instanced_attrib(2, np.zeros((1, 3), dtype=np.float32))
    _instanced_attrib(4, np.ones((1, 1), dtype=np.float32))

    ebo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, index_array.nbytes, index_array, gl.GL_STATIC_DRAW)
    gl.glBindVertexArray(0)

    return vao, len(index_array), colour_vbo, tint, n_surface


def _cone_mesh(radius: float, segments: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Unit-height cone as a triangle soup. Scaled per instance in the shader."""
    verts: list[tuple[float, float, float]] = []
    for s in range(segments):
        a0 = 2.0 * np.pi * s / segments
        a1 = 2.0 * np.pi * (s + 1) / segments
        verts.append((radius * np.cos(a0), 0.0, radius * np.sin(a0)))
        verts.append((radius * np.cos(a1), 0.0, radius * np.sin(a1)))
        verts.append((0.0, 1.0, 0.0))
    positions = np.asarray(verts, dtype=np.float32)
    triangles = np.arange(len(positions), dtype=np.uint32).reshape(-1, 3)
    return positions, _vertex_normals(positions, triangles)


def _build_instanced_cones(n_instances: int, radius: float):
    """Instanced cone VAO. Returns (vao, vertex_count, offset, colour, scale)."""
    import OpenGL.GL as gl

    positions, normals = _cone_mesh(radius)
    vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vao)
    _static_attrib(0, positions)
    _static_attrib(1, normals)
    offset = _instanced_attrib(2, np.zeros((n_instances, 3), dtype=np.float32))
    colour = _instanced_attrib(3, np.zeros((n_instances, 3), dtype=np.float32))
    scale = _instanced_attrib(4, np.zeros((n_instances, 1), dtype=np.float32))
    gl.glBindVertexArray(0)
    return vao, len(positions), offset, colour, scale


def _build_screen_quad(left: float, bottom: float, right: float, top: float):
    """Two counter-clockwise triangles in NDC. CCW matters: culling stays on."""
    import OpenGL.GL as gl

    verts = np.asarray(
        [
            (left, bottom, 0.0, 0.0),
            (right, bottom, 1.0, 0.0),
            (right, top, 1.0, 1.0),
            (left, bottom, 0.0, 0.0),
            (right, top, 1.0, 1.0),
            (left, top, 0.0, 1.0),
        ],
        dtype=np.float32,
    )
    vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vao)
    vbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, verts.nbytes, verts, gl.GL_STATIC_DRAW)
    stride = 4 * 4
    gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, None)
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(8))
    gl.glEnableVertexAttribArray(1)
    gl.glBindVertexArray(0)
    return vao


def _static_attrib(location: int, data: np.ndarray):
    import OpenGL.GL as gl

    vbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_STATIC_DRAW)
    gl.glVertexAttribPointer(location, data.shape[1], gl.GL_FLOAT, gl.GL_FALSE, 0, None)
    gl.glEnableVertexAttribArray(location)
    return vbo


def _instanced_attrib(location: int, data: np.ndarray):
    import OpenGL.GL as gl

    vbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)
    gl.glVertexAttribPointer(location, data.shape[1], gl.GL_FLOAT, gl.GL_FALSE, 0, None)
    gl.glEnableVertexAttribArray(location)
    gl.glVertexAttribDivisor(location, 1)
    return vbo


# ---------------------------------------------------------------------------
# maths
# ---------------------------------------------------------------------------


def _perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.radians(fov_y) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def _look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = center - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up / np.linalg.norm(up))
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = (-np.dot(s, eye), -np.dot(u, eye), np.dot(f, eye))
    return m


def _lerp_rgb(a, b, t: float) -> tuple[float, float, float]:
    t = min(max(t, 0.0), 1.0)
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


# ---------------------------------------------------------------------------
# renderer
# ---------------------------------------------------------------------------


class BlockRenderer:
    """Draws the four terrace zones, crop, weather and HUD."""

    def __init__(self, render_mode: str, size: tuple[int, int] = WINDOW_SIZE) -> None:
        import OpenGL.GL as gl
        import pygame

        self.render_mode = render_mode
        self.size = size
        self._closed = False

        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, GL_MAJOR)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, GL_MINOR)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
        )
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)

        flags = pygame.OPENGL | pygame.DOUBLEBUF
        if render_mode == "rgb_array":
            flags |= pygame.HIDDEN
        pygame.display.set_mode(size, flags)
        if render_mode == "human":
            pygame.display.set_caption("Umurima — block operations")

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_MULTISAMPLE)

        self._mesh_program = _link_program(MESH_VERTEX_SHADER, MESH_FRAGMENT_SHADER)
        self._sky_program = _link_program(SCREEN_VERTEX_SHADER, SKY_FRAGMENT_SHADER)
        self._hud_program = _link_program(SCREEN_VERTEX_SHADER, HUD_FRAGMENT_SHADER)

        (
            self._terrain_vao,
            self._terrain_indices,
            self._terrain_colours,
            self._terrain_tint,
            self._terrain_surface_vertices,
        ) = _build_terrain()

        crop = _build_instanced_cones(4 * CROP_PER_ZONE, CROP_RADIUS)
        self._crop_vao, self._crop_vertices = crop[0], crop[1]
        self._crop_offsets, self._crop_colours, self._crop_scales = crop[2], crop[3], crop[4]

        rain = _build_instanced_cones(RAIN_INSTANCES, 0.013)
        self._rain_vao, self._rain_vertices = rain[0], rain[1]
        self._rain_offsets, self._rain_colours, self._rain_scales = rain[2], rain[3], rain[4]

        self._sky_vao = _build_screen_quad(-1.0, -1.0, 1.0, 1.0)
        self._hud_vao = self._build_hud_quad()
        self._hud_texture = gl.glGenTextures(1)
        self._font = None
        self._font_bold = None

        self._azimuth, self._elevation, self._distance = CAMERA_START
        self._dragging = False
        self._view_proj = self._view_projection()

        # Placement is fixed for the whole episode so plants do not teleport
        # between frames; only height and colour track the zone state.
        self._crop_xz = self._crop_row_positions()
        self._rain_seed = self._rain_start_positions()
        self._start_ticks = pygame.time.get_ticks()

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def _build_hud_quad(self):
        w, h = self.size
        left = -1.0 + 2.0 * HUD_MARGIN / w
        right = left + 2.0 * HUD_SIZE[0] / w
        top = 1.0 - 2.0 * HUD_MARGIN / h
        bottom = top - 2.0 * HUD_SIZE[1] / h
        return _build_screen_quad(left, bottom, right, top)

    def _view_projection(self) -> np.ndarray:
        proj = _perspective(40.0, self.size[0] / self.size[1], 0.5, 80.0)
        target = np.asarray(CAMERA_TARGET, dtype=np.float32)
        az, el = np.radians(self._azimuth), np.radians(self._elevation)
        offset = np.array(
            [
                self._distance * np.cos(el) * np.sin(az),
                self._distance * np.sin(el),
                self._distance * np.cos(el) * np.cos(az),
            ],
            dtype=np.float32,
        )
        view = _look_at(target + offset, target, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        return (proj @ view).astype(np.float32)

    def _crop_row_positions(self) -> np.ndarray:
        """Plants laid out in rows along each bench, with a little jitter."""
        rng = np.random.default_rng(11)
        rows: list[tuple[float, float]] = []
        for zone in range(4):
            zone_start = zone * ZONE_DEPTH
            for crop_row in range(CROP_ROWS_PER_ZONE):
                # Keep clear of the bund on the downslope edge.
                fraction = 0.26 + 0.21 * crop_row
                z_centre = zone_start + ZONE_DEPTH * fraction
                for slot in range(CROP_PER_ROW):
                    x = 0.38 + (TOTAL_WIDTH - 0.76) * slot / (CROP_PER_ROW - 1)
                    rows.append(
                        (
                            x + float(rng.uniform(-0.045, 0.045)),
                            z_centre + float(rng.uniform(-0.035, 0.035)),
                        )
                    )
        return np.asarray(rows, dtype=np.float32)

    def _rain_start_positions(self) -> np.ndarray:
        rng = np.random.default_rng(23)
        x = rng.uniform(-0.6, TOTAL_WIDTH + 0.6, RAIN_INSTANCES)
        z = rng.uniform(-0.6, TOTAL_DEPTH + 0.6, RAIN_INSTANCES)
        phase = rng.uniform(0.0, RAIN_SPAN, RAIN_INSTANCES)
        speed = rng.uniform(0.85, 1.20, RAIN_INSTANCES)
        return np.stack([x, z, phase, speed], axis=1).astype(np.float32)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def closed(self) -> bool:
        return self._closed

    def draw(self, state: dict[str, Any]) -> np.ndarray | None:
        """Render one frame. Returns an (H, W, 3) uint8 array in rgb_array mode."""
        import OpenGL.GL as gl
        import pygame

        if self._closed:
            return None
        if self.render_mode == "human" and not self._pump_events():
            return None

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        raining = float(state.get("rain_today_mm", 0.0)) > 0.5
        self._draw_sky(raining)
        self._draw_scene(state, raining)
        self._draw_hud(state)

        if self.render_mode == "human":
            pygame.display.flip()
            return None

        raw = gl.glReadPixels(0, 0, self.size[0], self.size[1], gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(self.size[1], self.size[0], 3)
        return np.flipud(frame).copy()

    def close(self) -> None:
        """Idempotent, called more than once in practice."""
        import pygame

        if self._closed:
            return
        self._closed = True
        pygame.quit()

    # ------------------------------------------------------------------
    # interaction
    # ------------------------------------------------------------------

    def _pump_events(self) -> bool:
        """Handle window and camera input. Returns False once the view is closed."""
        import pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.close()
                    return False
                if event.key == pygame.K_r:
                    self._azimuth, self._elevation, self._distance = CAMERA_START
                    self._view_proj = self._view_projection()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._dragging = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self._dragging = False
            elif event.type == pygame.MOUSEMOTION and self._dragging:
                dx, dy = event.rel
                self._azimuth = (self._azimuth - dx * ORBIT_SENSITIVITY) % 360.0
                self._elevation = _clamp(
                    self._elevation + dy * ORBIT_SENSITIVITY, *ELEVATION_RANGE
                )
                self._view_proj = self._view_projection()
            elif event.type == pygame.MOUSEWHEEL:
                self._distance = _clamp(
                    self._distance * (1.0 - event.y * ZOOM_SENSITIVITY), *DISTANCE_RANGE
                )
                self._view_proj = self._view_projection()
        return True

    def _elapsed_seconds(self) -> float:
        """Wall-clock time, so animation is independent of the policy step rate."""
        import pygame

        return (pygame.time.get_ticks() - self._start_ticks) / 1000.0

    # ------------------------------------------------------------------
    # passes
    # ------------------------------------------------------------------

    def _draw_sky(self, raining: bool) -> None:
        import OpenGL.GL as gl

        zenith = SKY_ZENITH_WET_RGB if raining else SKY_ZENITH_RGB
        horizon = SKY_HORIZON_WET_RGB if raining else SKY_HORIZON_RGB
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(gl.GL_FALSE)
        gl.glUseProgram(self._sky_program)
        gl.glUniform3f(gl.glGetUniformLocation(self._sky_program, "u_zenith"), *zenith)
        gl.glUniform3f(gl.glGetUniformLocation(self._sky_program, "u_horizon"), *horizon)
        gl.glBindVertexArray(self._sky_vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glEnable(gl.GL_DEPTH_TEST)

    def _draw_scene(self, state: dict[str, Any], raining: bool) -> None:
        import OpenGL.GL as gl

        program = self._mesh_program
        gl.glUseProgram(program)
        gl.glUniformMatrix4fv(
            gl.glGetUniformLocation(program, "u_view_proj"), 1, gl.GL_TRUE, self._view_proj
        )
        gl.glUniform3f(gl.glGetUniformLocation(program, "u_light_dir"), 0.55, -1.0, 0.42)
        light = (0.76, 0.80, 0.86) if raining else (1.0, 0.96, 0.88)
        gl.glUniform3f(gl.glGetUniformLocation(program, "u_light_colour"), *light)
        unlit = gl.glGetUniformLocation(program, "u_unlit")
        gl.glUniform1f(unlit, 0.0)

        self._upload_terrain_colours(state)
        gl.glBindVertexArray(self._terrain_vao)
        gl.glDrawElementsInstanced(
            gl.GL_TRIANGLES, self._terrain_indices, gl.GL_UNSIGNED_INT, None, 1
        )

        self._upload_crop(state)
        gl.glBindVertexArray(self._crop_vao)
        gl.glDrawArraysInstanced(gl.GL_TRIANGLES, 0, self._crop_vertices, 4 * CROP_PER_ZONE)

        drops = self._upload_rain(state)
        if drops:
            # Rain is drawn unlit so drops stay bright against dark wet soil.
            gl.glUniform1f(unlit, 1.0)
            gl.glBindVertexArray(self._rain_vao)
            gl.glDrawArraysInstanced(gl.GL_TRIANGLES, 0, self._rain_vertices, drops)
            gl.glUniform1f(unlit, 0.0)

    def _upload_terrain_colours(self, state: dict[str, Any]) -> None:
        import OpenGL.GL as gl

        wetness = _zone_wetness(state)
        colours = np.zeros((len(self._terrain_tint), 3), dtype=np.float32)
        stride = TERRAIN_COLS + 1
        for row in range(TERRAIN_ROWS + 1):
            rgb = _lerp_rgb(SOIL_DRY_RGB, SOIL_WET_RGB, wetness[_zone_of_row(row)])
            colours[row * stride : (row + 1) * stride] = rgb
        # The extruded sides are subsoil, not topsoil, so they stay constant.
        colours[self._terrain_surface_vertices :] = SUBSOIL_RGB
        colours *= self._terrain_tint[:, None]
        np.clip(colours, 0.0, 1.0, out=colours)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._terrain_colours)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, colours.nbytes, colours)

    def _upload_crop(self, state: dict[str, Any]) -> None:
        import OpenGL.GL as gl

        zones = state.get("zones", [])
        harvested = float(state.get("harvested_fraction", 0.0))
        total = 4 * CROP_PER_ZONE
        offsets = np.zeros((total, 3), dtype=np.float32)
        colours = np.zeros((total, 3), dtype=np.float32)
        scales = np.zeros((total, 1), dtype=np.float32)
        lifted_per_zone = int(round(harvested * CROP_PER_ZONE))

        for zone_index in range(4):
            zone = zones[zone_index] if zone_index < len(zones) else {}
            canopy = float(zone.get("canopy_cover", 0.0))
            nitrogen = float(zone.get("nitrogen_kg_ha", 60.0))
            damage = float(zone.get("pest_damage", 0.0))
            health = min(nitrogen / 70.0, 1.0) * (1.0 - 0.6 * damage)
            colour = _lerp_rgb(CROP_STRESSED_RGB, CROP_HEALTHY_RGB, health)
            bench_y = ZONE_HEIGHTS[zone_index]

            start = zone_index * CROP_PER_ZONE
            for slot in range(CROP_PER_ZONE):
                idx = start + slot
                x, z = self._crop_xz[idx]
                offsets[idx] = (x, bench_y, z)
                colours[idx] = colour
                # Harvest clears plants progressively rather than all at once.
                picked = slot < lifted_per_zone
                scales[idx] = 0.0 if picked else canopy * CROP_MAX_HEIGHT + 0.03

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._crop_offsets)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, offsets.nbytes, offsets)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._crop_colours)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, colours.nbytes, colours)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._crop_scales)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, scales.nbytes, scales)

    def _upload_rain(self, state: dict[str, Any]) -> int:
        """Returns how many drops to draw, scaled by today's rainfall."""
        import OpenGL.GL as gl

        rain_mm = float(state.get("rain_today_mm", 0.0))
        if rain_mm <= 0.5:
            return 0
        drops = int(min(1.0, rain_mm / 22.0) * RAIN_INSTANCES)
        if drops == 0:
            return 0

        seed = self._rain_seed[:drops]
        elapsed = self._elapsed_seconds()
        # Continuous fall driven by wall-clock time, wrapped into the fall span,
        # so drops stream smoothly instead of jumping once per simulated day.
        fallen = (seed[:, 2] + elapsed * RAIN_SPEED * seed[:, 3]) % RAIN_SPAN

        offsets = np.empty((drops, 3), dtype=np.float32)
        offsets[:, 0] = seed[:, 0]
        offsets[:, 1] = RAIN_TOP - fallen
        offsets[:, 2] = seed[:, 1]
        colours = np.tile(np.asarray(RAIN_RGB, dtype=np.float32), (drops, 1))
        scales = np.full((drops, 1), 0.34, dtype=np.float32)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._rain_offsets)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, offsets.nbytes, offsets)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._rain_colours)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, colours.nbytes, colours)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._rain_scales)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, scales.nbytes, scales)
        return drops

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------

    def _draw_hud(self, state: dict[str, Any]) -> None:
        import OpenGL.GL as gl

        self._upload_hud_texture(state)
        gl.glUseProgram(self._hud_program)
        gl.glUniform1i(gl.glGetUniformLocation(self._hud_program, "u_texture"), 0)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glBindVertexArray(self._hud_vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_DEPTH_TEST)

    def _upload_hud_texture(self, state: dict[str, Any]) -> None:
        import OpenGL.GL as gl
        import pygame

        surface = self._build_hud(state)
        data = pygame.image.tobytes(surface, "RGBA", True)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._hud_texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, HUD_SIZE[0], HUD_SIZE[1], 0,
            gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

    def _build_hud(self, state: dict[str, Any]):
        """Text overlay. Pre-harvest interval line goes red inside the window,
        and the flowering window is called out because stress there is permanent."""
        import pygame

        if self._font is None:
            pygame.font.init()
            self._font = pygame.font.SysFont("menlo,monaco,monospace", 15)
            self._font_bold = pygame.font.SysFont("menlo,monaco,monospace", 16, bold=True)
            self._font_small = pygame.font.SysFont("menlo,monaco,monospace", 12)

        surface = pygame.Surface(HUD_SIZE, pygame.SRCALPHA)
        surface.fill((14, 20, 28, 194))
        pygame.draw.rect(surface, (150, 170, 190, 90), surface.get_rect(), width=1)

        flowering = bool(state.get("flowering", False))
        within_phi = bool(state.get("within_phi", False))
        forecast = list(state.get("rain_forecast_mm", []))[:3]
        forecast += [0.0] * (3 - len(forecast))

        # Header: the stage turns amber during flowering because water stress in
        # that window costs yield that later irrigation cannot recover.
        prefix = (
            f"{state.get('block_id', '?')}  Day {int(state.get('day', 0)):3d}"
            f"/{int(state.get('horizon', 120))}  {state.get('season', '?')}  "
        )
        surface.blit(self._font_bold.render(prefix, True, HUD_BRIGHT_RGB), (12, 8))
        surface.blit(
            self._font_bold.render(
                _stage_name(state.get("stage", 0.0)),
                True,
                HUD_AMBER_RGB if flowering else HUD_BRIGHT_RGB,
            ),
            (12 + self._font_bold.size(prefix)[0], 8),
        )

        wet_ahead = sum(forecast) > 5.0
        rows = [
            (
                f"Cash {state.get('cash_krwf', 0.0):+8.0f} kRWF"
                f"   Reservoir {state.get('reservoir_fraction', 0.0):4.0%}",
                HUD_TEXT_RGB,
            ),
            (
                f"Return {state.get('return', 0.0):+8.2f}"
                f"   Price {state.get('price_rwf', 0.0):5.0f} RWF/kg",
                HUD_TEXT_RGB,
            ),
            (
                f"Yield {state.get('yield_forecast_kg', 0.0):8.0f} kg"
                f"   Picked {state.get('harvested_fraction', 0.0):8.0%}",
                HUD_TEXT_RGB,
            ),
            (
                f"Rain {state.get('rain_today_mm', 0.0):6.1f} mm"
                f"   Stress {float(state.get('water_stress', 0.0)):8.0%}",
                HUD_TEXT_RGB,
            ),
            (
                f"Forecast {forecast[0]:4.1f} /{forecast[1]:5.1f} /{forecast[2]:5.1f} mm",
                HUD_BLUE_RGB if wet_ahead else HUD_TEXT_RGB,
            ),
            (f"Action {state.get('action', 'IDLE')}", HUD_TEXT_RGB),
        ]

        alerts: list[tuple[str, tuple[int, int, int]]] = []
        if flowering:
            alerts.append(("Flowering: stress now is permanent", HUD_AMBER_RGB))
        if within_phi:
            alerts.append(("PHI violation: consignment rejected", HUD_RED_RGB))
        if not alerts:
            alerts.append(("PHI clear", HUD_GREEN_RGB))

        y = 34
        for text, colour in rows + alerts:
            surface.blit(self._font.render(text, True, colour), (12, y))
            y += 21

        self._draw_zone_panel(surface, state)

        if self.render_mode == "human":
            surface.blit(
                self._font.render("drag orbit · scroll zoom · R reset", True, HUD_DIM_RGB),
                (12, HUD_SIZE[1] - 21),
            )
        return surface

    def _draw_zone_panel(self, surface, state: dict[str, Any]) -> None:
        """Four rows, one per terrace, each a small bar chart of that bench's state.

        Water reads as how full the profile is, so a long bar is a wet bench and
        the agent irrigating a short bar is visibly the right call.
        """
        import pygame

        zones = state.get("zones", [])
        if not zones:
            return
        font = self._font_small
        x0 = ZONE_PANEL_X

        pygame.draw.line(
            surface, (70, 84, 98), (x0 - 16, 26), (x0 - 16, 26 + 4 * ZONE_ROW_H), 1
        )
        for i, (name, _) in enumerate(ZONE_PANEL_BARS):
            surface.blit(
                font.render(name, True, HUD_DIM_RGB),
                (x0 + 84 + i * (ZONE_BAR_W + ZONE_BAR_GAP), 12),
            )

        for row in range(4):
            zone = zones[row] if row < len(zones) else {}
            y = 32 + row * ZONE_ROW_H
            surface.blit(font.render(ZONE_LABELS[row], True, HUD_TEXT_RGB), (x0, y))

            water = 1.0 - float(zone.get("depletion_frac", 0.0))
            values = (
                water,
                float(zone.get("canopy_cover", 0.0)),
                float(zone.get("pest_pressure", 0.0)),
                float(zone.get("weed_pressure", 0.0)),
            )
            for i, ((_, colour), value) in enumerate(
                zip(ZONE_PANEL_BARS, values, strict=True)
            ):
                bx = x0 + 84 + i * (ZONE_BAR_W + ZONE_BAR_GAP)
                by = y + 4
                pygame.draw.rect(surface, (48, 58, 70), (bx, by, ZONE_BAR_W, ZONE_BAR_H))
                filled = int(ZONE_BAR_W * min(max(value, 0.0), 1.0))
                if filled > 0:
                    pygame.draw.rect(surface, colour, (bx, by, filled, ZONE_BAR_H))
                pygame.draw.rect(
                    surface, (86, 98, 112), (bx, by, ZONE_BAR_W, ZONE_BAR_H), 1
                )


def _zone_wetness(state: dict[str, Any]) -> list[float]:
    """Depletion as a 0..1 wetness fraction per zone, for soil colouring.

    Uses the fraction of the zone's own available water, not absolute mm. A deep
    valley bench holds several times the water of a thin ridge bench, so a fixed
    mm divisor makes every terrace shade almost identically and the soil reads as
    one flat colour no matter what the agent does.
    """
    zones = state.get("zones", [])
    wetness = []
    for index in range(4):
        zone = zones[index] if index < len(zones) else {}
        if "depletion_frac" in zone:
            frac = float(zone["depletion_frac"])
        else:  # older state dicts
            frac = min(float(zone.get("depletion_mm", 0.0)) / 90.0, 1.0)
        wetness.append(1.0 - min(max(frac, 0.0), 1.0))
    return wetness


def _stage_name(stage: float) -> str:
    """Map the quantised stage value back to its label.

    _stage_value emits 0, 1/3, 2/3, 1, so this rounds rather than truncates:
    int(0.66 * 3) is 1, which would label the flowering stage "Vegetative".
    """
    names = ("Establishment", "Vegetative", "Flowering", "Ripening")
    return names[min(int(round(stage * 3.0)), 3)]
