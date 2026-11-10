# Audited AST fixture extracted from frozen BEHAVIOR-1K 6a6194600.
STRUCTURE_CATEGORIES = frozenset({'floors', 'walls', 'ceilings', 'lawn', 'driveway', 'fence', 'roof', 'background'})
GROUND_CATEGORIES = frozenset({'floors', 'lawn', 'driveway', 'carpet'})

class InteractiveTraversableScene:

    def _should_load_object(self, obj_info, task_metadata):
        agent_ok = super()._should_load_object(obj_info, task_metadata)
        name = obj_info['args']['name']
        category = obj_info['args'].get('category', 'object')
        in_rooms = obj_info['args'].get('in_rooms', [])
        if isinstance(in_rooms, str):
            assert ',' not in in_rooms
        in_rooms = [in_rooms] if isinstance(in_rooms, str) else in_rooms
        not_blacklisted = self.not_load_object_categories is None or category not in self.not_load_object_categories
        task_relevant_names = set(task_metadata['inst_to_name'].values()) if 'inst_to_name' in task_metadata else set()
        is_task_relevant = name in task_relevant_names or category in STRUCTURE_CATEGORIES
        whitelisted = self.load_object_categories is None and (not self.load_task_relevant_only) or (self.load_object_categories is not None and category in self.load_object_categories) or (self.load_task_relevant_only and is_task_relevant)
        valid_room = self.load_room_instances is None or len(set(self.load_room_instances) & set(in_rooms)) > 0
        is_building_structure = category in STRUCTURE_CATEGORIES - GROUND_CATEGORIES or category in ['door', 'sliding_door']
        return not_blacklisted and whitelisted and valid_room and agent_ok or is_building_structure
