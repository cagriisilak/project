# battery.py
class BatterySystem:
    def __init__(self, modules_config):
        """
        Simplified: Track total energy, modules only for swap counting
        """

        #print(f"DEBUG: Initializing battery with {len(modules_config)} modules")

        import copy
        self.modules = copy.deepcopy(modules_config)
        
        # Ensure all charges are between 0 and 100
        for module in self.modules:
            module['charge'] = max(0.0, min(100.0, module['charge']))
        
        self.total_capacity_kwh = sum(m['capacity'] for m in self.modules)
        self.total_charge_kwh = sum(m['capacity'] * (m['charge'] / 100.0) 
                                for m in self.modules)

        #for i, module in enumerate(self.modules):
        #    print(f"  Module {i}: capacity={module['capacity']}kWh, charge={module['charge']}%")
        
        #print(f"  Total capacity: {self.total_capacity_kwh}kWh")
        #print(f"  Total charge: {self.total_charge_kwh}kWh")
        #print(f"  Initial SOC: {self.get_total_soc():.1f}%")

        self.swapped_modules_count = 0
        
    def get_total_soc(self):
        """Get total state of charge (0-100%)"""
        if self.total_capacity_kwh == 0:
            return 0
        return (self.total_charge_kwh / self.total_capacity_kwh) * 100
    
    def consume_energy(self, energy_kwh):
        """
        Consume energy SEQUENTIALLY from modules as per PDF spec
        Returns: True if successful, False if not enough energy
        """
        remaining_energy = energy_kwh
    
        # Drain modules in sequence
        for i, module in enumerate(self.modules):
            if remaining_energy <= 0:
                break
            
            # Calculate how much energy is in this module
            module_energy = module['capacity'] * (module['charge'] / 100.0)
        
            if module_energy >= remaining_energy:
                # This module has enough energy
                module_energy -= remaining_energy
                remaining_energy = 0
            else:
                # Drain this module completely
                remaining_energy -= module_energy
                module_energy = 0
        
            # Update module charge percentage
            module['charge'] = (module_energy / module['capacity']) * 100
    
        # Update total charge
        self.total_charge_kwh = sum(m['capacity'] * (m['charge'] / 100.0) for m in self.modules)
    
        # Return True if we had enough energy
        return remaining_energy == 0
    
    def get_available_energy_kwh(self):
        """Get total available energy"""
        return self.total_charge_kwh
    
    def swap_modules(self, module_indices=None):
        """
        Swap specified modules (or all if None)
        Returns: (swap_time_minutes, swap_cost)
        """
        if module_indices is None:
            # Swap all modules that are depleted (< 20%)
            module_indices = []
            for i, module in enumerate(self.modules):
                if module['charge'] < 20.0:
                    module_indices.append(i)
        
        swap_time = 0
        swap_cost = 0
        
        for idx in module_indices:
            if 0 <= idx < len(self.modules):
                module = self.modules[idx]
                
                # Update total charge
                module_energy = module['capacity'] * 1.0  # 100% = 1.0
                old_module_energy = module['capacity'] * (module['charge'] / 100.0)
                self.total_charge_kwh += (module_energy - old_module_energy)

                # Reset to 100%
                module['charge'] = 100.0
                
                # Add to swapped count
                self.swapped_modules_count += 1
                
                # Cost and time
                swap_time += 2  # 2 minutes per module
                swap_cost += 50  # 50 cost per module
        
        return swap_time, swap_cost
    
    def needs_swap(self, threshold_percent=20.0):
        """Check if any module needs swapping"""
        for module in self.modules:
            if 0 < module['charge'] < threshold_percent:
                return True
        return False