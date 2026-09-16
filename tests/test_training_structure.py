import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parsed(relative_path: str) -> ast.Module:
    return ast.parse((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))


def call_count(tree: ast.AST, name: str) -> int:
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    )


def is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    comparison = node.test
    return (
        isinstance(comparison.left, ast.Name)
        and comparison.left.id == "__name__"
        and len(comparison.comparators) == 1
        and isinstance(comparison.comparators[0], ast.Constant)
        and comparison.comparators[0].value == "__main__"
    )


class TrainingStructureTests(unittest.TestCase):
    def test_training_has_one_loop_and_one_ppo_update_call_site(self):
        tree = parsed("training/ddppo.py")
        top_level_functions = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }

        self.assertIn("run_training_loop", top_level_functions)
        self.assertEqual(call_count(tree, "run_training_loop"), 1)
        self.assertEqual(call_count(tree, "perform_ppo_update"), 1)
        self.assertNotIn("perform_ppo_update_single_gpu", top_level_functions)
        self.assertNotIn("perform_ppo_update_multi_gpu", top_level_functions)

    def test_ppo_builds_selected_features_once_and_reuses_done_prefix(self):
        tree = parsed("training/ddppo.py")
        update = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "perform_ppo_update"
        )
        selected_cache_calls = [
            node
            for node in ast.walk(update)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "build_selected_feature_cache"
        ]
        cumsum_calls = [
            node
            for node in ast.walk(update)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "cumsum"
        ]

        self.assertEqual(len(selected_cache_calls), 1)
        self.assertEqual(len(cumsum_calls), 1)

    def test_rollout_loop_has_no_per_step_distributed_boolean_sync(self):
        tree = parsed("training/ddppo.py")
        loop = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_training_loop"
        )
        all_true_calls = [
            node
            for node in ast.walk(loop)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "all_true"
        ]

        self.assertEqual(all_true_calls, [])

    def test_production_modules_do_not_embed_manual_main_blocks(self):
        production_files = [
            PROJECT_ROOT / "training" / "ddppo.py",
            PROJECT_ROOT / "training" / "network.py",
            *sorted((PROJECT_ROOT / "simulator").glob("*.py")),
        ]
        offenders = []
        for path in production_files:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            if any(is_main_guard(node) for node in tree.body):
                offenders.append(str(path.relative_to(PROJECT_ROOT)))

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
