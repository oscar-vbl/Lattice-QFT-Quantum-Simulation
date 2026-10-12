import pytest

from QuantumSimulation.SchwingerSimulation import SchwingerSimulation


@pytest.mark.integration
def test_initial_state_pipeline_without_dynamics(simple_config, staggered_2):
    sim = SchwingerSimulation(simple_config, initial_state=staggered_2)
    sim.run_simulation()
    assert sim.initial_state is not None
    assert sim.vacuum_energy is not None
    assert sim.ground_state_diagnostics["vacuum_energy"] == sim.vacuum_energy


@pytest.mark.integration
def test_vqe_pipeline_without_dynamics(simple_config):
    sim = SchwingerSimulation(simple_config)
    sim.run_simulation(initial_vqe_params=None)
    assert sim.vqe_result is not None
    assert sim.initial_state is not None
    assert sim.vacuum_parameters is not None
