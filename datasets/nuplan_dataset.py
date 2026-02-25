import os
from typing import Callable, List, Optional, Tuple, Union
import random
from joblib import Parallel, delayed
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario

from datasets import get_scenario_map
from datasets import get_filter_parameters
from datasets import get_features
from datasets import get_plan_scenario_types

DEFAULT_SAVE_DIR = '/root/wangjunyi/data/planr1'


class NuplanDataset(Dataset):
    def __init__(self,
                 root: str,
                 dir:str,
                 split: str,
                 mode: str,
                 transform: Optional[Callable] = None,
                 historical_horizon: float = 2,
                 future_horizon:int = 8,
                 num_samples_per_second: int = 10,
                 num_total_scenarios: int = 1000000,
                 ratio: float = 0.1,
                 parallel: bool=True,
                 save_dir: Optional[str] = None) -> None:

        self.root = root
        self.save_dir = save_dir if save_dir is not None else DEFAULT_SAVE_DIR
        if dir in ['train', 'val', 'test', 'mini']:
            self.dir = dir
        else:
            raise ValueError(dir + ' is not valid')
        if split in ['train', 'val']:
            self.split = split
        else:
            raise ValueError(split + ' is not valid')
        self.mode = mode
        if mode not in ['pred', 'plan']:
            raise ValueError(mode + ' is not valid')
        
        self.map_version = "nuplan-maps-v1.0"
        self.map_path = os.path.join(self.root, 'nuplan-v1.1', 'maps')
        self.limit_total_scenarios = num_total_scenarios
        
        self._raw_file_names = os.listdir(os.path.join(self.root, 'nuplan-v1.1', 'splits', self.dir))

        self.processed_file_names_path = os.path.join(self.save_dir, f"{self.dir}-processed_file_names-{self.mode}-{self.split}-PlanR1.pt")
        print(f"NuplanDataset: loading {self.split} list from {self.processed_file_names_path}")
        if os.path.exists(self.processed_file_names_path):
            self._processed_file_names = torch.load(self.processed_file_names_path)
            print(f"Number of scenarios in {self.split} dataset: {len(self._processed_file_names)}")
        else:
            print(f"File not found. Building scenario list from nuPlan API (may take 30min+), then saving to {self.processed_file_names_path}")
            self._processed_file_names = []
            scenario_mapping = ScenarioMapping(scenario_map=get_scenario_map(), subsample_ratio_override=0.5)
            if self.mode == 'plan':
                scenario_types = get_plan_scenario_types()
                scenario_filter = ScenarioFilter(*get_filter_parameters(limit_total_scenarios=self.limit_total_scenarios, scenario_types=scenario_types))
            elif self.mode == 'pred':
                scenario_filter = ScenarioFilter(*get_filter_parameters(limit_total_scenarios=self.limit_total_scenarios))
            worker = SingleMachineParallelExecutor(use_process_pool=True)
            builder = NuPlanScenarioBuilder(self.raw_paths, self.map_path, None, None, self.map_version, scenario_mapping=scenario_mapping)
            scenarios = builder.get_scenarios(scenario_filter, worker)
            print(f"Number of total scenarios: {len(scenarios)}")
            for scenario in tqdm(scenarios):
                scenario_type = scenario.scenario_type
                scenario_name = scenario.scenario_name
                self._processed_file_names.append(f"{scenario_type}-{scenario_name}.pt")
            random.seed(42)
            random.shuffle(self._processed_file_names)
            os.makedirs(self.save_dir, exist_ok=True)
            torch.save(self._processed_file_names[:int(self.limit_total_scenarios*ratio)], os.path.join(self.save_dir, f"{self.dir}-processed_file_names-{self.mode}-val-PlanR1.pt"))
            torch.save(self._processed_file_names[int(self.limit_total_scenarios*ratio):], os.path.join(self.save_dir, f"{self.dir}-processed_file_names-{self.mode}-train-PlanR1.pt"))
            worker._executor.shutdown(wait=True)

        # 只算一次目录，避免 90 万次属性访问导致卡顿
        base_dir = os.path.join(self.save_dir, f"{self.dir}-processed-{self.mode}-{self.split}-PlanR1")
        num_names = len(self._processed_file_names)
        print(f"Building path list for {num_names} samples...")
        self._processed_paths = [os.path.join(base_dir, name) for name in self._processed_file_names]
        print(f"Path list built.")

        self.num_samples_per_second = num_samples_per_second
        self.historical_horizon = historical_horizon
        self.num_historical_steps = int(historical_horizon * num_samples_per_second)
        self.future_horizon = future_horizon
        self.num_future_steps = int(future_horizon * num_samples_per_second)
        self.parallel = parallel

        print(f"NuplanDataset [{self.split}] calling parent __init__...", flush=True)
        super(NuplanDataset, self).__init__(root=root, transform=transform)
        print(f"NuplanDataset [{self.split}] 已加载完成，共 {len(self._processed_file_names)} 条样本。", flush=True)

    def _process(self) -> None:
        # 数据已通过文件名列表加载，跳过父类对 processed_paths 的 90 万次 files_exist 检查
        return

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'nuplan-v1.1', 'splits', self.dir)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.save_dir, f"{self.dir}-processed-{self.mode}-{self.split}-PlanR1")
    
    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def process(self) -> None:
        scenario_mapping = ScenarioMapping(scenario_map=get_scenario_map(), subsample_ratio_override=0.5)
        if self.mode == 'plan':
            scenario_types = get_plan_scenario_types()
            scenario_filter = ScenarioFilter(*get_filter_parameters(limit_total_scenarios=self.limit_total_scenarios, scenario_types=scenario_types))
        elif self.mode == 'pred':
            scenario_filter = ScenarioFilter(*get_filter_parameters(limit_total_scenarios=self.limit_total_scenarios))
        worker = SingleMachineParallelExecutor(use_process_pool=True)
        builder = NuPlanScenarioBuilder(self.raw_paths, self.map_path, None, None, self.map_version, scenario_mapping=scenario_mapping)
        scenarios = builder.get_scenarios(scenario_filter, worker)

        os.makedirs(os.path.join(self.save_dir, f"{self.dir}-processed-{self.mode}-train-PlanR1"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, f"{self.dir}-processed-{self.mode}-val-PlanR1"), exist_ok=True)
        self.train_file_names = torch.load(os.path.join(self.save_dir, f"{self.dir}-processed_file_names-{self.mode}-train-PlanR1.pt"))
        self.val_file_names = torch.load(os.path.join(self.save_dir, f"{self.dir}-processed_file_names-{self.mode}-val-PlanR1.pt"))
                
        if self.parallel:
            batch_size = 50
            process_map(self.process_batch_scenario, 
                        [scenarios[i:i+batch_size] for i in range(0, len(scenarios), batch_size)],
                        max_workers=100, 
                        chunksize=1)
        else:
            for scenario in tqdm(scenarios):
                self.process_single_scenario(scenario)

    def process_batch_scenario(self, batch: List[NuPlanScenario]) -> None:
        """
        Process a batch of scenarios to reduce overhead.
        """
        for scenario in batch:
            self.process_single_scenario(scenario)

    def process_single_scenario(self, scenario: NuPlanScenario) -> None:
        scenario_type = scenario.scenario_type
        scenario_name = scenario.scenario_name

        data = dict()
        data['log_name'] = scenario.log_name
        data['scenario_type'] = scenario_type
        data['scenario_name'] = scenario_name

        # get features
        present_ego_state = scenario.initial_ego_state
        past_ego_state = list(scenario.get_ego_past_trajectory(iteration=0, num_samples=self.num_historical_steps, time_horizon=self.historical_horizon))
        future_ego_state = list(scenario.get_ego_future_trajectory(iteration=0, num_samples=self.num_future_steps, time_horizon=self.future_horizon))
        ego_state_buffer = past_ego_state + [present_ego_state] + future_ego_state

        present_observation = scenario.initial_tracked_objects
        past_observation = list(scenario.get_past_tracked_objects(iteration=0, num_samples=self.num_historical_steps, time_horizon=self.historical_horizon))
        future_observation = list(scenario.get_future_tracked_objects(iteration=0, num_samples=self.num_future_steps, time_horizon=self.future_horizon))
        observation_buffer = past_observation + [present_observation] + future_observation

        map_api = scenario.map_api
        traffic_lights = scenario.get_traffic_light_status_at_iteration(iteration=0)
        route_roadblock_ids = scenario.get_route_roadblock_ids()
        
        data.update(get_features(ego_state_buffer, observation_buffer, map_api, traffic_lights, route_roadblock_ids, max_agents=20))

        if f"{scenario_type}-{scenario_name}.pt" in self.train_file_names:
            torch.save(data, os.path.join(self.save_dir, f"{self.dir}-processed-{self.mode}-train-PlanR1", f"{scenario_type}-{scenario_name}.pt"))
        elif f"{scenario_type}-{scenario_name}.pt" in self.val_file_names:
            torch.save(data, os.path.join(self.save_dir, f"{self.dir}-processed-{self.mode}-val-PlanR1", f"{scenario_type}-{scenario_name}.pt"))
        else:
            raise ValueError(f"{scenario_type}-{scenario_name}.pt is not in train or val")
        
    def len(self) -> int:
        return len(self.processed_file_names)

    def get(self, idx: int) -> HeteroData:     
        return HeteroData(torch.load(self.processed_paths[idx]))