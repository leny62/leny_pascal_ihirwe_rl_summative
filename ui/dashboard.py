"""Umurima block operations dashboard.

    uv run --extra ui streamlit run ui/dashboard.py
"""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from ui._manager import EpisodeManager

# ---------------------------------------------------------------------------
# page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Umurima Block Monitor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# session state initialisation
# ---------------------------------------------------------------------------

DEFAULT_SEED = 42
MODEL_GLOB = "models/**/*.zip"


def _init_session() -> None:
    if "manager" not in st.session_state:
        st.session_state.manager = None
    if "auto_play" not in st.session_state:
        st.session_state.auto_play = False


_init_session()

# ---------------------------------------------------------------------------
# sidebar — controls
# ---------------------------------------------------------------------------


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Controls")

        policy_options = ["scripted", "random"]
        for p in sorted(Path("models").glob("**/*.zip")):
            policy_options.append(str(p))
        for p in sorted(Path("models").glob("**/*.pt")):
            policy_options.append(str(p))

        policy = st.selectbox("Policy", policy_options, index=0, key="policy_select")
        seed = st.number_input("Seed", min_value=0, max_value=9999, value=DEFAULT_SEED, key="seed_input")

        col1, col2 = st.columns(2)
        if col1.button("New episode", width="stretch", type="primary"):
            st.session_state.manager = EpisodeManager(seed=int(seed), policy_spec=policy)
            st.session_state.auto_play = False
            st.rerun()

        if col2.button("Reset", width="stretch"):
            st.session_state.manager = None
            st.session_state.auto_play = False
            st.rerun()

        mgr = st.session_state.manager
        if mgr is not None and not mgr.done:
            st.divider()
            days = st.number_input("Days to advance", min_value=1, max_value=50, value=1, key="days_input")
            c1, c2 = st.columns(2)
            if c1.button("Step", width="stretch"):
                records = mgr.step(int(days))
                if records:
                    for r in records:
                        st.toast(f"Day {r.day}: {r.action} ({r.reward:+.3f})")
                if mgr.done:
                    st.toast(f"Episode ended: {mgr.termination_cause}", icon="🏁")
                st.rerun()

            auto = c2.toggle("Auto-play", key="auto_toggle")
            if auto != st.session_state.auto_play:
                st.session_state.auto_play = auto
                if auto:
                    st.rerun()

        if mgr is not None and mgr.done:
            st.success(f"Finished: {mgr.termination_cause}")


# ---------------------------------------------------------------------------
# main area
# ---------------------------------------------------------------------------


def _render_dashboard(mgr: EpisodeManager) -> None:
    state = mgr.state
    _render_kpi_row(state, mgr)
    _render_zone_grid(state)
    _render_detail_tabs(mgr, state)


def _render_kpi_row(state: dict, mgr: EpisodeManager) -> None:
    day, horizon = state.get("day", 0), state.get("horizon", 120)
    cash = state.get("cash_krwf", 0)
    reservoir = state.get("reservoir_fraction", 0)
    total_return = state.get("return", 0)
    yield_kg = state.get("yield_forecast_kg", 0)
    harvested = state.get("harvested_fraction", 0)
    rain = state.get("rain_today_mm", 0)
    stage_val = state.get("stage", 0)
    stage_names = ["Establishment", "Vegetative", "Flowering", "Ripening"]
    # Rounded, not truncated: the env emits 0, 1/3, 2/3, 1 and int(0.66 * 3)
    # is 1, which would label the flowering stage "Vegetative".
    stage = stage_names[min(int(round(stage_val * 3)), 3)]

    cols = st.columns(8)
    cols[0].metric("Day", f"{day}/{horizon}", delta=stage, delta_color="off")
    cols[1].metric("Cash (kRWF)", f"{cash:+.0f}", delta=None)
    cols[2].metric("Reservoir", f"{reservoir:.0%}")
    cols[3].metric("Return", f"{total_return:+.2f}")
    cols[4].metric("Yield est.", f"{yield_kg:.0f} kg")
    cols[5].metric("Harvested", f"{harvested:.0%}")
    cols[6].metric("Rain today", f"{rain:.1f} mm")
    phi = state.get("within_phi", False)
    cols[7].metric("PHI", "⚠️ VIOLATION" if phi else "clear", delta=None)

    st.progress(day / horizon, text=f"Season progress — {stage}")

    if cash < -80:
        st.warning(f"Cash critically low ({cash:+.0f} kRWF). Insolvency at -100.")
    if reservoir < 0.15:
        st.warning("Reservoir critically low.")


def _render_zone_grid(state: dict) -> None:
    zones = state.get("zones", [])
    if not zones:
        return

    cols = st.columns(len(zones))
    for i, (col, zone) in enumerate(zip(cols, zones, strict=True)):
        with col:
            # Fraction of this zone's own available water, not a fixed mm scale.
            # Total available water happens to be 84 mm on every bench today,
            # because the bean root zone caps at 0.60 m and all four are deeper,
            # but normalising per zone keeps this correct if the rooting depth
            # or the soil depths ever change.
            depletion_frac = float(
                zone.get("depletion_frac", min(zone.get("depletion_mm", 0) / 100.0, 1.0))
            )
            canopy = zone.get("canopy_cover", 0)
            nitrogen = zone.get("nitrogen_kg_ha", 0)
            pest = zone.get("pest_pressure", 0)

            depletion_color = _depletion_color(depletion_frac)
            canopy_color = _canopy_color(canopy)
            nitrogen_color = _nitrogen_color(nitrogen)

            st.markdown(f"**Zone {i}** (ridge → valley)" if i == 0 else f"**Zone {i}**")
            st.markdown(
                f"<span style='color:{depletion_color};font-size:18px'>●</span> "
                f"Moisture: {1 - depletion_frac:.0%}",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<span style='color:{canopy_color};font-size:18px'>●</span> "
                f"Canopy: {canopy:.1%}",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<span style='color:{nitrogen_color};font-size:18px'>●</span> "
                f"N: {nitrogen:.0f} kg/ha",
                unsafe_allow_html=True,
            )
            st.caption(
                f"Pest {pest:.0%} · Depth {zone.get('soil_depth_m', 0):.2f} m · "
                f"Weed {zone.get('weed_pressure', 0):.1%}"
            )


def _render_detail_tabs(mgr: EpisodeManager, state: dict) -> None:
    tab1, tab2, tab3, tab4 = st.tabs(["Action log", "Ledger", "Zone details", "Weather"])

    with tab1:
        history = mgr.history
        if not history:
            st.caption("No actions yet. Step forward to begin.")
        else:
            import pandas as pd

            df = pd.DataFrame(
                [
                    {"Day": r.day, "Action": r.action, "Reward": r.reward, "Cash (kRWF)": r.cash}
                    for r in history
                ]
            )
            st.dataframe(
                df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Reward": st.column_config.NumberColumn(format="%+.4f"),
                    "Cash (kRWF)": st.column_config.NumberColumn(format="%+.2f"),
                },
            )

    with tab2:
        ledger = mgr.ledger
        events = ledger.get("events", [])
        totals = ledger.get("totals", {})
        head = ledger.get("head", "?")

        st.caption(f"Chain head: `{head[:24]}...` · Verified: {_verified_badge(mgr)}")

        tc = st.columns(6)
        tc[0].metric("Water", f"{totals.get('water_mm', 0):.1f} mm")
        tc[1].metric("N applied", f"{totals.get('nitrogen_kg_ha', 0):.0f} kg/ha")
        tc[2].metric("K applied", f"{totals.get('potash_kg_ha', 0):.0f} kg/ha")
        tc[3].metric("Sprays", f"{totals.get('sprays', 0):.0f}")
        tc[4].metric("Weeding", f"{totals.get('weeding_events', 0):.0f}")
        tc[5].metric("Revenue", f"{totals.get('harvest_revenue_krwf', 0):.1f} kRWF")

        if events:
            import pandas as pd

            st.dataframe(
                pd.DataFrame(events),
                width="stretch",
                hide_index=True,
            )

    with tab3:
        zones = state.get("zones", [])
        if zones:
            import pandas as pd

            zone_df = pd.DataFrame(
                [
                    {
                        "Zone": z["index"],
                        "Soil depth (m)": f"{z['soil_depth_m']:.2f}",
                        "Depletion (mm)": f"{z['depletion_mm']:.1f}",
                        "Canopy": f"{z['canopy_cover']:.3f}",
                        "Nitrogen (kg/ha)": f"{z['nitrogen_kg_ha']:.1f}",
                        "Pest pressure": f"{z['pest_pressure']:.3f}",
                        "Pest damage": f"{z['pest_damage']:.3f}",
                        "Weed pressure": f"{z['weed_pressure']:.3f}",
                    }
                    for z in zones
                ]
            )
            st.dataframe(zone_df, width="stretch", hide_index=True)

    with tab4:
        forecast = state.get("rain_forecast_mm", [])
        wcols = st.columns(4)
        wcols[0].metric("Rain (today)", f"{state.get('rain_today_mm', 0):.1f} mm")
        wcols[1].metric("Rain +1d", f"{forecast[0]:.1f} mm" if len(forecast) > 0 else "?")
        wcols[2].metric("Rain +2d", f"{forecast[1]:.1f} mm" if len(forecast) > 1 else "?")
        wcols[3].metric("Rain +3d", f"{forecast[2]:.1f} mm" if len(forecast) > 2 else "?")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _depletion_color(frac: float) -> str:
    if frac < 0.3:
        return "#2e86c1"
    if frac < 0.6:
        return "#f0b27a"
    return "#e74c3c"


def _canopy_color(frac: float) -> str:
    if frac < 0.15:
        return "#d4ac0d"
    if frac < 0.40:
        return "#7dcea0"
    return "#1e8449"


def _nitrogen_color(kg_ha: float) -> str:
    if kg_ha < 20:
        return "#e74c3c"
    if kg_ha < 50:
        return "#f0b27a"
    return "#1e8449"


def _verified_badge(mgr: EpisodeManager) -> str:
    return "✅" if mgr.ledger_verified else "❌"


# ---------------------------------------------------------------------------
# auto-play loop
# ---------------------------------------------------------------------------


def _auto_play() -> None:
    mgr = st.session_state.manager
    if mgr is None or mgr.done or not st.session_state.auto_play:
        return
    time.sleep(0.1)
    mgr.step(1)
    if mgr.done:
        st.session_state.auto_play = False
        st.toast(f"Episode ended: {mgr.termination_cause}", icon="🏁")
    st.rerun()


# ---------------------------------------------------------------------------
# main render
# ---------------------------------------------------------------------------


_render_sidebar()

mgr = st.session_state.manager
if mgr is None:
    st.markdown(
        "## Umurima Block Monitor\n\n"
        "Select a policy and seed in the sidebar, then press **New episode** to begin."
    )
    with st.expander("About"):
        st.markdown(
            "Simulates a 120-day french bean growing season across four terrace "
            "zones on a Rwandan hillside. The agent chooses one of 18 actions each "
            "day: irrigate (light or heavy, per-zone or whole-block), apply nitrogen "
            "or potash, spray biopesticide, hire weeding crews, scout, or harvest."
        )
else:
    _render_dashboard(mgr)
    if st.session_state.auto_play:
        _auto_play()
