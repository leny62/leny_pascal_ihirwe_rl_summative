"""PyOpenGL renderer for the block.

Imported lazily from UmurimaEnv.render so training never opens a GL context.

macOS caps at OpenGL 4.1 core with GLSL 4.10 over Metal. Shaders target
#version 410 core. The core profile must be requested explicitly via
pygame GL attributes before creating the window, otherwise the default
legacy 2.1 context is returned. No compute shaders.
"""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np

WINDOW_SIZE = (1280, 720)
GL_MAJOR, GL_MINOR = 3, 3

SOIL_DRY_RGB = (0.72, 0.58, 0.36)
SOIL_WET_RGB = (0.28, 0.20, 0.13)
CROP_HEALTHY_RGB = (0.22, 0.55, 0.18)
CROP_STRESSED_RGB = (0.78, 0.74, 0.26)

VERTEX_SHADER = """#version 410 core
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

FRAGMENT_SHADER = """#version 410 core
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

HUD_VERTEX = """#version 410 core
layout (location = 0) in vec2 in_pos;
layout (location = 1) in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
}
"""

HUD_FRAGMENT = """#version 410 core
in vec2 v_uv;
uniform sampler2D u_texture;
out vec4 frag_colour;
void main() {
    frag_colour = texture(u_texture, v_uv);
}
"""

TERRAIN_COLS = 40
TERRAIN_ROWS = 20
CROP_INSTANCES_PER_ZONE = 12
CROP_MAX_HEIGHT = 0.35
ZONE_HEIGHTS = [1.2, 0.8, 0.5, 0.2]
TOTAL_WIDTH = 8.0
ZONE_DEPTH = 1.0
TOTAL_DEPTH = 4.0


def _compile_shader(src: str, shader_type: int) -> Any:
    from OpenGL.GL import glCompileShader, glCreateShader, glGetShaderInfoLog, glShaderSource

    shader = glCreateShader(shader_type)
    glShaderSource(shader, src)
    glCompileShader(shader)
    log = glGetShaderInfoLog(shader)
    if log and log.strip():
        raise RuntimeError(f"shader compile error: {log.decode()}")
    return shader


def _link_program(vertex_src: str, fragment_src: str) -> Any:
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


def _build_terrain_mesh():
    """Grid mesh for four terrace benches. Returns (vao, index_count, colour_vbo)."""
    import OpenGL.GL as gl

    verts: list[float] = []
    colours: list[float] = []
    indices: list[int] = []

    for row in range(TERRAIN_ROWS + 1):
        z = (row / TERRAIN_ROWS) * TOTAL_DEPTH
        zone_idx = min(int(z / ZONE_DEPTH), 3)
        y = ZONE_HEIGHTS[zone_idx]
        for col in range(TERRAIN_COLS + 1):
            x = (col / TERRAIN_COLS) * TOTAL_WIDTH
            verts.extend([x, y, z])
            colours.extend(SOIL_DRY_RGB)

    for row in range(TERRAIN_ROWS):
        for col in range(TERRAIN_COLS):
            tl = row * (TERRAIN_COLS + 1) + col
            tr = tl + 1
            bl = (row + 1) * (TERRAIN_COLS + 1) + col
            br = bl + 1
            indices.extend([tl, bl, tr, tr, bl, br])

    verts_arr = np.array(verts, dtype=np.float32)
    colours_arr = np.array(colours, dtype=np.float32)

    normals = np.zeros((len(verts) // 3, 3), dtype=np.float32)
    idx_arr = np.array(indices, dtype=np.uint32).reshape(-1, 3)
    for tri in idx_arr:
        a, b, c = verts_arr.reshape(-1, 3)[tri]
        n = np.cross(b - a, c - a)
        for i in tri:
            normals[i] += n
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals = (normals / norms).flatten()

    vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vao)

    vbo_pos = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_pos)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, verts_arr.nbytes, verts_arr, gl.GL_STATIC_DRAW)
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
    gl.glEnableVertexAttribArray(0)

    vbo_norm = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_norm)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, normals.nbytes, normals, gl.GL_STATIC_DRAW)
    gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
    gl.glEnableVertexAttribArray(1)

    colour_vbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, colour_vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, colours_arr.nbytes, colours_arr, gl.GL_DYNAMIC_DRAW)
    gl.glVertexAttribPointer(3, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
    gl.glEnableVertexAttribArray(3)
    gl.glVertexAttribDivisor(3, 0)

    offset_buf = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, offset_buf)
    offset_data = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, offset_data.nbytes, offset_data, gl.GL_STATIC_DRAW)
    gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
    gl.glEnableVertexAttribArray(2)
    gl.glVertexAttribDivisor(2, 1)

    scale_buf = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, scale_buf)
    scale_data = np.array([1.0], dtype=np.float32)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, scale_data.nbytes, scale_data, gl.GL_STATIC_DRAW)
    gl.glVertexAttribPointer(4, 1, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
    gl.glEnableVertexAttribArray(4)
    gl.glVertexAttribDivisor(4, 1)

    ebo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
    idx_arr = np.array(indices, dtype=np.uint32)
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, idx_arr.nbytes, idx_arr, gl.GL_STATIC_DRAW)

    gl.glBindVertexArray(0)
    return vao, len(indices), colour_vbo


def _build_crop_instances():
    """Unit cone for crop stalks. Returns (vao, vertex_count, offset_vbo, colour_vbo, scale_vbo)."""
    import OpenGL.GL as gl

    r = 0.03
    h = 1.0
    segments = 6
    verts: list[float] = []
    for i in range(segments):
        angle = 2.0 * np.pi * i / segments
        angle2 = 2.0 * np.pi * (i + 1) / segments
        x1, z1 = r * np.cos(angle), r * np.sin(angle)
        x2, z2 = r * np.cos(angle2), r * np.sin(angle2)
        verts.extend([x1, 0.0, z1, x2, 0.0, z2, 0.0, h, 0.0])
    verts_arr = np.array(verts, dtype=np.float32)

    v_count = len(verts) // 3
    normals = np.zeros((v_count, 3), dtype=np.float32)
    for i in range(0, v_count, 3):
        a, b, c = verts_arr.reshape(-1, 3)[i : i + 3]
        n = np.cross(b - a, c - a)
        normals[i : i + 3] = n
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals = (normals / norms).flatten()

    vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vao)

    vbo_pos = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_pos)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, verts_arr.nbytes, verts_arr, gl.GL_STATIC_DRAW)
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
    gl.glEnableVertexAttribArray(0)

    vbo_norm = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_norm)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, normals.nbytes, normals, gl.GL_STATIC_DRAW)
    gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
    gl.glEnableVertexAttribArray(1)

    n_instances = 4 * CROP_INSTANCES_PER_ZONE
    offset_buf = gl.glGenBuffers(1)
    colour_buf = gl.glGenBuffers(1)
    scale_buf = gl.glGenBuffers(1)

    for buf, loc, comps in [
        (offset_buf, 2, 3),
        (colour_buf, 3, 3),
        (scale_buf, 4, 1),
    ]:
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buf)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, n_instances * comps * 4, None, gl.GL_DYNAMIC_DRAW)
        gl.glVertexAttribPointer(loc, comps, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(loc)
        gl.glVertexAttribDivisor(loc, 1)

    gl.glBindVertexArray(0)
    return vao, v_count, offset_buf, colour_buf, scale_buf


def _build_hud_quad(screen_w: int, screen_h: int, hud_w: int, hud_h: int):
    """Quad positioned in the top-left corner for the HUD overlay."""
    import OpenGL.GL as gl

    l = -1.0
    r = -1.0 + 2.0 * hud_w / screen_w
    t = 1.0
    b = 1.0 - 2.0 * hud_h / screen_h
    verts = np.array(
        [l, b, 0, 0, r, b, 1, 0, l, t, 0, 1, r, b, 1, 0, l, t, 0, 1, r, t, 1, 1],
        dtype=np.float32,
    )
    vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vao)
    vbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, verts.nbytes, verts, gl.GL_STATIC_DRAW)
    gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 16, None)
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 16, ctypes.c_void_p(8))
    gl.glEnableVertexAttribArray(1)
    gl.glBindVertexArray(0)
    return vao


def _lerp_rgb(a: tuple[float, ...], b: tuple[float, ...], t: float) -> tuple[float, float, float]:
    t = max(0.0, min(1.0, t))
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def _perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.radians(fov_y) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def _look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = center - eye
    f /= np.linalg.norm(f)
    u = up / np.linalg.norm(up)
    s = np.cross(f, u)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[:3, 3] = [-np.dot(s, eye), -np.dot(u, eye), np.dot(f, eye)]
    return m


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
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, 1)

        flags = pygame.OPENGL | pygame.DOUBLEBUF
        if render_mode == "rgb_array":
            flags |= pygame.HIDDEN
        pygame.display.set_mode(size, flags)
        if render_mode == "human":
            pygame.display.set_caption("Umurima")

        gl.glClearColor(0.45, 0.65, 0.85, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)

        self._program = _link_program(VERTEX_SHADER, FRAGMENT_SHADER)
        self._hud_program = _link_program(HUD_VERTEX, HUD_FRAGMENT)

        self._terrain_vao, self._terrain_indices, self._terrain_colour_vbo = _build_terrain_mesh()
        crop_vao, crop_verts, off_buf, col_buf, scl_buf = _build_crop_instances()
        self._crop_vao = crop_vao
        self._crop_vertex_count = crop_verts
        self._crop_offset_buf = off_buf
        self._crop_colour_buf = col_buf
        self._crop_scale_buf = scl_buf

        self._hud_vao = _build_hud_quad(size[0], size[1], 520, 90)
        self._hud_texture = gl.glGenTextures(1)

        self._font = None
        self._view_proj = self._compute_view_proj()

    def _compute_view_proj(self) -> np.ndarray:
        aspect = self.size[0] / self.size[1]
        proj = _perspective(45.0, aspect, 0.5, 30.0)
        eye = np.array([4.0, 3.5, -2.5], dtype=np.float32)
        center = np.array([4.0, 0.6, 2.0], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        view = _look_at(eye, center, up)
        return (proj @ view).astype(np.float32)

    def draw(self, state: dict[str, Any]) -> np.ndarray | None:
        import OpenGL.GL as gl
        import pygame

        if self._closed:
            return None
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glUseProgram(self._program)

        vp_loc = gl.glGetUniformLocation(self._program, "u_view_proj")
        gl.glUniformMatrix4fv(vp_loc, 1, gl.GL_TRUE, self._view_proj)
        light_loc = gl.glGetUniformLocation(self._program, "u_light_dir")
        gl.glUniform3f(light_loc, 0.6, -1.0, 0.4)
        lc_loc = gl.glGetUniformLocation(self._program, "u_light_colour")
        gl.glUniform3f(lc_loc, 1.0, 0.95, 0.85)

        self._update_terrain_colours(state)
        gl.glBindVertexArray(self._terrain_vao)
        gl.glDrawElementsInstanced(gl.GL_TRIANGLES, self._terrain_indices, gl.GL_UNSIGNED_INT, None, 1)

        self._update_crop_instances(state)
        gl.glBindVertexArray(self._crop_vao)
        n_instances = 4 * CROP_INSTANCES_PER_ZONE
        gl.glDrawArraysInstanced(gl.GL_TRIANGLES, 0, self._crop_vertex_count, n_instances)

        gl.glUseProgram(self._hud_program)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        self._update_hud(state)
        gl.glBindVertexArray(self._hud_vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)

        if self.render_mode == "human":
            pygame.display.flip()
            return None

        frame = gl.glReadPixels(0, 0, self.size[0], self.size[1], gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        arr = np.frombuffer(frame, dtype=np.uint8).reshape(self.size[1], self.size[0], 3)
        return np.flipud(arr).copy()

    def _update_terrain_colours(self, state: dict[str, Any]) -> None:
        import OpenGL.GL as gl

        zones = state.get("zones", [])
        colours = np.zeros(((TERRAIN_ROWS + 1) * (TERRAIN_COLS + 1), 3), dtype=np.float32)
        idx = 0
        for row in range(TERRAIN_ROWS + 1):
            z = (row / TERRAIN_ROWS) * TOTAL_DEPTH
            zi = min(int(z / ZONE_DEPTH), 3)
            if zi < len(zones):
                zone = zones[zi]
                frac = np.clip(zone.get("depletion_mm", 0) / 100.0, 0.0, 1.0)
            else:
                frac = 0.0
            rgb = _lerp_rgb(SOIL_DRY_RGB, SOIL_WET_RGB, frac)
            colours[idx : idx + TERRAIN_COLS + 1] = rgb
            idx += TERRAIN_COLS + 1

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._terrain_colour_vbo)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, colours.nbytes, colours.flatten())

    def _update_crop_instances(self, state: dict[str, Any]) -> None:
        import OpenGL.GL as gl

        zones = state.get("zones", [])
        n = 4 * CROP_INSTANCES_PER_ZONE
        offsets = np.zeros((n, 3), dtype=np.float32)
        colours = np.zeros((n, 3), dtype=np.float32)
        scales = np.zeros(n, dtype=np.float32)

        rng = np.random.default_rng(42)
        idx = 0
        for zi in range(4):
            zone = zones[zi] if zi < len(zones) else {}
            canopy = float(zone.get("canopy_cover", 0))
            n_stress = float(np.clip(1.0 - zone.get("nitrogen_kg_ha", 50) / 100.0, 0.0, 1.0))
            colour = _lerp_rgb(CROP_STRESSED_RGB, CROP_HEALTHY_RGB, 1.0 - n_stress)
            base_z = zi * ZONE_DEPTH + 0.5
            for _ in range(CROP_INSTANCES_PER_ZONE):
                x = float(rng.uniform(0.3, TOTAL_WIDTH - 0.3))
                z = float(rng.uniform(base_z - 0.35, base_z + 0.35))
                offsets[idx] = [x, ZONE_HEIGHTS[zi], z]
                colours[idx] = colour
                scales[idx] = canopy * CROP_MAX_HEIGHT + 0.02
                idx += 1

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._crop_offset_buf)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, offsets.nbytes, offsets)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._crop_colour_buf)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, colours.nbytes, colours)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._crop_scale_buf)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, scales.nbytes, scales)

    def _update_hud(self, state: dict[str, Any]) -> None:
        import OpenGL.GL as gl
        import pygame

        if self._font is None:
            pygame.font.init()
            self._font = pygame.font.SysFont("monospace", 15, bold=True)

        day = state.get("day", 0)
        horizon = state.get("horizon", 120)
        season = state.get("season", "?")
        stage_val = state.get("stage", 0.0)
        stage_names = ["Establishment", "Vegetative", "Flowering", "Ripening"]
        stage_idx = min(int(stage_val * 3), 3)
        stage = stage_names[stage_idx]
        cash = state.get("cash_krwf", 0)
        reservoir = state.get("reservoir_fraction", 0)
        action = state.get("action", "IDLE")
        total_return = state.get("return", 0)
        harvested = state.get("harvested_fraction", 0)
        rain = state.get("rain_today_mm", 0)
        phi = state.get("within_phi", False)
        yield_kg = state.get("yield_forecast_kg", 0)

        lines = [
            f"Day {day:3d}/{horizon}  Season {season}  {stage}",
            f"Cash: {cash:+.0f} kRWF  Reservoir: {reservoir:.0%}  Rain: {rain:.1f} mm",
            f"Action: {action}  Return: {total_return:+.2f}  Harvested: {harvested:.0%}",
            f"Yield: {yield_kg:.0f} kg  PHI: {'VIOLATION' if phi else 'clear'}",
        ]

        hud_w, hud_h = 520, 90
        surf = pygame.Surface((hud_w, hud_h), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 150))
        y = 4
        for line in lines:
            text = self._font.render(line, True, (220, 220, 220))
            surf.blit(text, (8, y))
            y += 20

        data = pygame.image.tobytes(surf, "RGBA", True)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._hud_texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, hud_w, hud_h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

    def close(self) -> None:
        import pygame

        if self._closed:
            return
        self._closed = True
        pygame.quit()
