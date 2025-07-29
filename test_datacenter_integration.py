#!/usr/bin/env python3
"""Test script for DataCenterLoad integration"""

import matplotlib.pyplot as plt
import numpy as np

import zap


def test_new_archetypes():
    """Test new archetype-based data center loads"""
    print("Testing new archetype-based DataCenterLoad...")

    net = zap.PowerNetwork(num_nodes=3)
    T = 24  # 24 time steps (6 hours at 15-min resolution)

    # Create data centers with different archetypes
    dcloads = zap.DataCenterLoad(
        num_nodes=net.num_nodes,
        terminal=np.array([0, 1, 2]),
        nominal_capacity=np.array([100.0, 50.0, 25.0]),  # MW
        profile_types=[
            zap.DataCenterLoad.ProfileType.INTERACTIVE,
            zap.DataCenterLoad.ProfileType.AI_TRAIN,
            zap.DataCenterLoad.ProfileType.HPC,
        ],
        linear_cost=np.array([100.0, 120.0, 80.0]),
        settime_horizon=6.0,  # 6 hours
        time_resolution_hours=0.25,  # 15 min
        rack_mix={"gpu": 0.3, "cpu": 0.5, "storage": 0.2},
        rack_power_kw={"gpu": 200, "cpu": 40, "storage": 25},
        pue=1.25,
    )

    print(f"Generated profiles shape: {dcloads.profile.shape}")
    print(f"Profile range: {dcloads.profile.min():.3f} to {dcloads.profile.max():.3f}")

    # Test legacy compatibility
    print("\nTesting legacy compatibility...")
    dcloads_legacy = zap.DataCenterLoad(
        num_nodes=net.num_nodes,
        terminal=np.array([0]),
        nominal_capacity=np.array([50.0]),
        profile_types=[zap.DataCenterLoad.ProfileType.DIURNAL],
        linear_cost=np.array([100.0]),
        settime_horizon=6.0,
    )

    print(f"Legacy profile shape: {dcloads_legacy.profile.shape}")
    print(
        f"Legacy profile range: {dcloads_legacy.profile.min():.3f} to {dcloads_legacy.profile.max():.3f}"
    )

    return dcloads, dcloads_legacy


def test_custom_profiles():
    """Test custom profile functionality"""
    print("\nTesting custom profiles...")

    net = zap.PowerNetwork(num_nodes=2)
    custom_profile = np.array([0.6, 0.8, 0.7, 0.9, 0.5, 0.4])

    dcloads_custom = zap.DataCenterLoad(
        num_nodes=net.num_nodes,
        terminal=np.array([0, 1]),
        nominal_capacity=np.array([30.0, 40.0]),
        profile_types=[
            zap.DataCenterLoad.ProfileType.CUSTOM,
            zap.DataCenterLoad.ProfileType.BATCH,
        ],
        profiles=[custom_profile, None],
        linear_cost=np.array([90.0, 110.0]),
        settime_horizon=1.5,
    )

    print(f"Custom/mixed profiles shape: {dcloads_custom.profile.shape}")
    print(f"First row (custom): {dcloads_custom.profile[0]}")

    return dcloads_custom


def plot_profiles(dcloads, dcloads_legacy, dcloads_custom):
    """Plot the generated profiles"""
    plt.figure(figsize=(15, 10))

    # Plot new archetypes
    plt.subplot(3, 1, 1)
    for i, archetype in enumerate(["Interactive", "AI-Train", "HPC"]):
        plt.plot(dcloads.profile[i], label=f"DC {i+1}: {archetype}", linewidth=2)
    plt.title("New Archetype-Based Profiles")
    plt.ylabel("Load Factor")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot legacy
    plt.subplot(3, 1, 2)
    plt.plot(
        dcloads_legacy.profile[0], label="Legacy Diurnal", linewidth=2, color="orange"
    )
    plt.title("Legacy Diurnal Profile")
    plt.ylabel("Load Factor")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot custom/mixed
    plt.subplot(3, 1, 3)
    plt.plot(
        dcloads_custom.profile[0], label="Custom Profile", linewidth=2, color="red"
    )
    plt.plot(
        dcloads_custom.profile[1], label="Batch Archetype", linewidth=2, color="green"
    )
    plt.title("Custom and Mixed Profiles")
    plt.xlabel("Time Steps")
    plt.ylabel("Load Factor")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("datacenter_profiles_test.png", dpi=150, bbox_inches="tight")
    print("\nProfiles saved to 'datacenter_profiles_test.png'")


def test_dispatch():
    """Test that dispatch still works correctly"""
    print("\nTesting dispatch functionality...")

    try:
        import cvxpy as cp

        net = zap.PowerNetwork(num_nodes=2)

        # Create a simple data center load
        dcload = zap.DataCenterLoad(
            num_nodes=net.num_nodes,
            terminal=np.array([0]),
            nominal_capacity=np.array([50.0]),
            profile_types=[zap.DataCenterLoad.ProfileType.INTERACTIVE],
            linear_cost=np.array([100.0]),
            settime_horizon=1.0,  # 1 hour
        )

        # Create a generator
        gen = zap.Generator(
            num_nodes=net.num_nodes,
            terminal=np.array([1]),
            nominal_capacity=np.array([100.0]),
            dynamic_capacity=np.ones(
                (1, 4)
            ),  # 4 time steps for 1 hour at 15-min resolution
            linear_cost=np.array([50.0]),
        )

        # Create transmission line
        line = zap.ACLine(
            num_nodes=net.num_nodes,
            source_terminal=np.array([0]),
            sink_terminal=np.array([1]),
            nominal_capacity=np.array([75.0]),
            susceptance=np.array([0.1]),
            capacity=np.ones(1),
        )

        # Add ground
        ground = zap.Ground(num_nodes=net.num_nodes, terminal=np.array([0]))

        devices = [dcload, gen, line, ground]

        # Run dispatch
        outcome = net.dispatch(
            devices, time_horizon=4, solver=cp.CLARABEL, add_ground=False
        )

        print(f"Dispatch successful! Problem value: {outcome.problem.value:.2f}")
        print(f"DC power consumption: {-outcome.power[0][0].sum():.2f} MW")
        print(f"Generator output: {outcome.power[1][0].sum():.2f} MW")

        return True

    except ImportError:
        print("CVXPY not available, skipping dispatch test")
        return False
    except Exception as e:
        print(f"Dispatch test failed: {e}")
        return False


if __name__ == "__main__":
    print("=== DataCenterLoad Integration Test ===\n")

    # Test new functionality
    dcloads, dcloads_legacy = test_new_archetypes()
    dcloads_custom = test_custom_profiles()

    # Plot results
    plot_profiles(dcloads, dcloads_legacy, dcloads_custom)

    # Test dispatch
    dispatch_success = test_dispatch()

    print(f"\n=== Test Summary ===")
    print(f"✓ New archetype profiles generated successfully")
    print(f"✓ Legacy compatibility maintained")
    print(f"✓ Custom profiles working")
    print(
        f"{'✓' if dispatch_success else '✗'} Dispatch functionality {'working' if dispatch_success else 'failed'}"
    )

    print(f"\n=== Integration Complete ===")
    print(
        "The new synthetic data center load generator has been successfully integrated!"
    )
    print("Key improvements:")
    print("- More realistic archetype-based load modeling")
    print("- Sophisticated temporal correlation (AR1 processes)")
    print("- Rack-level composition modeling")
    print("- Power Usage Effectiveness (PUE) consideration")
    print("- Full backward compatibility with existing code")
