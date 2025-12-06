import carb.input
import numpy as np
import omni
import omni.appwindow

class gamepad_cmd:
    def __init__(self,
                 base_command: np.array = np.array([0.0, 0.0, 0.0]),
                 # Default mapping for an Xbox/PS controller
                 # Maps GamepadInput enum -> Direction Vector
                 input_mapping: dict = None, 
                 deadzone: float = 0.05,
                 scale: float = 10.0
                ) -> None:
        
        self._base_command = base_command
        self._deadzone = deadzone
        self._scale = scale

        # Acquire interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        
        # Get the first gamepad (Index 0)
        self._gamepad = self._appwindow.get_gamepad(0)
        
        # Define default mapping if None provided
        if input_mapping is None:
            self._input_mapping = {
                # Left Stick Y (Surge)
                carb.input.GamepadInput.LEFT_STICK_UP:    np.array([1.0, 0.0, 0.0]),
                carb.input.GamepadInput.LEFT_STICK_DOWN:  np.array([-1.0, 0.0, 0.0]),
                # Left Stick X (Sway)
                carb.input.GamepadInput.LEFT_STICK_LEFT:  np.array([0.0, 1.0, 0.0]),
                carb.input.GamepadInput.LEFT_STICK_RIGHT: np.array([0.0, -1.0, 0.0]),
                # Triggers / Right Stick (Heave)
                carb.input.GamepadInput.RIGHT_STICK_UP:   np.array([0.0, 0.0, 1.0]),
                carb.input.GamepadInput.RIGHT_STICK_DOWN: np.array([0.0, 0.0, -1.0]),
            }
        else:
            self._input_mapping = input_mapping

        # Store current value of each input (for continuous control)
        self._current_values = {key: 0.0 for key in self._input_mapping}

        if self._gamepad:
            device_name = getattr(self._gamepad, "name", "Unknown Gamepad")
            print(f"[gamepad_cmd] Connected to gamepad: {device_name}")
            self._sub_gamepad = self._input.subscribe_to_gamepad_events(self._gamepad, self._sub_gamepad_event)
        else:
            print("[gamepad_cmd] No gamepad detected at index 0.")
            self._sub_gamepad = None

    def _sub_gamepad_event(self, event, *args, **kwargs) -> bool:
        """Callback for gamepad events."""
        if event.input in self._input_mapping:
            val = event.value
            # Apply Deadzone
            if abs(val) < self._deadzone:
                val = 0.0
            
            # Store the current normalized value (0.0 to 1.0)
            self._current_values[event.input] = val
            
            # Recalculate the total command based on all active inputs
            self._update_command()
            
        return True

    def _update_command(self):
        """Sum all active inputs to create the command vector."""
        cmd = np.array([0.0, 0.0, 0.0])
        for input_id, direction in self._input_mapping.items():
            val = self._current_values[input_id]
            cmd += direction * val * self._scale
        
        self._base_command = cmd

    def cleanup(self):
        if self._sub_gamepad:
            self._input.unsubscribe_to_input_events(self._sub_gamepad)
        self._sub_gamepad = None
        self._gamepad = None
        self._appwindow = None
        self._input = None