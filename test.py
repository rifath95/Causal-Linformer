import unittest  # [Codex comment] Use the Python standard library so the sanity checks need no extra test dependency.

import torch  # [Codex comment] Create deterministic tensors and inspect numerical attention behavior.

import model as model_module  # [Codex comment] Access the module-level attention selector for an integration test.
from model import Block, CausalBlockwiseLinformerAttention, MultiHeadAttention  # [Codex comment] Import the new module and baseline selector targets under test.


class CausalBlockwiseLinformerAttentionTests(unittest.TestCase):  # [Codex comment] Group focused correctness checks for the new attention operator.
    def setUp(self):  # [Codex comment] Create a small deterministic module matching the specification's toy dimensions.
        torch.manual_seed(7)  # [Codex comment] Make random inputs and initialized linear layers reproducible.
        self.attn = CausalBlockwiseLinformerAttention(  # [Codex comment] Instantiate a CPU-friendly attention module independently of training config.
            max_context_length=12,  # [Codex comment] Use the toy example's maximum sequence length.
            hidden_size=8,  # [Codex comment] Keep test tensors small while retaining multiple heads.
            num_heads=2,  # [Codex comment] Exercise per-head projection weights.
            window_size=3,  # [Codex comment] Match the requested three-token local window.
            global_blocks=3,  # [Codex comment] Split the maximum context into three fixed global blocks.
            share_projections=False,  # [Codex comment] Test the preferred per-head Linformer projection mode.
            attention_dropout=0.0,  # [Codex comment] Disable randomness so causality comparisons are exact up to tolerance.
        )  # [Codex comment] Finish construction of the reusable test module.
        self.attn.eval()  # [Codex comment] Keep the module deterministic during all checks.

    def test_output_shape_for_non_divisible_runtime_length(self):  # [Codex comment] Verify generation-like prefix lengths do not need block divisibility.
        x = torch.randn(2, 10, 8)  # [Codex comment] Use a runtime prefix with a trailing two-token partial block.
        output = self.attn(x)  # [Codex comment] Run both local and completed-global branches.
        self.assertEqual(output.shape, x.shape)  # [Codex comment] Preserve the drop-in [batch, time, hidden] shape contract.

    def test_global_mask_matches_inclusive_toy_pattern(self):  # [Codex comment] Check exactly when each completed block becomes visible.
        mask = self.attn._build_global_mask(12, 3, torch.device("cpu"))  # [Codex comment] Build the full toy mask for three four-token blocks.
        expected_counts = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3])  # [Codex comment] Encode the requested inclusive visibility count at every query.
        self.assertTrue(torch.equal(mask.sum(dim=-1), expected_counts))  # [Codex comment] Confirm block visibility starts at positions 3, 7, and 11.
        self.assertTrue(mask[3, 0])  # [Codex comment] Confirm the final token of block zero may attend to that completed block.
        self.assertFalse(mask[6, 1])  # [Codex comment] Confirm block one remains hidden one token before it completes.
        self.assertTrue(mask[7, 1])  # [Codex comment] Confirm block one becomes visible exactly at its final token.

    def test_partial_runtime_block_is_not_compressed(self):  # [Codex comment] Verify only complete fixed-size blocks enter the global branch.
        runtime_length = 10  # [Codex comment] Leave the third four-token block incomplete by two tokens.
        completed_blocks = runtime_length // self.attn.global_block_size  # [Codex comment] Apply the required runtime completed-block calculation.
        self.assertEqual(completed_blocks, 2)  # [Codex comment] Confirm only the first eight tokens form global summaries.
        mask = self.attn._build_global_mask(runtime_length, completed_blocks, torch.device("cpu"))  # [Codex comment] Build a mask containing only those two summaries.
        self.assertEqual(mask.shape, (10, 2))  # [Codex comment] Ensure no placeholder summary exists for the partial third block.

    def test_future_changes_do_not_affect_earlier_outputs(self):  # [Codex comment] Enforce strict autoregressive causality across both branches.
        x1 = torch.randn(1, 12, 8)  # [Codex comment] Create the reference full-length hidden-state sequence.
        x2 = x1.clone()  # [Codex comment] Keep both sequences identical through the comparison position.
        comparison_position = 7  # [Codex comment] Test an inclusive completed-block boundary where block one is globally visible.
        x2[:, comparison_position + 1:] = torch.randn_like(x2[:, comparison_position + 1:])  # [Codex comment] Change only tokens strictly after the causal boundary.
        output1 = self.attn(x1)  # [Codex comment] Compute outputs for the reference sequence.
        output2 = self.attn(x2)  # [Codex comment] Compute outputs after changing future-only inputs.
        self.assertTrue(torch.allclose(output1[:, :comparison_position + 1], output2[:, :comparison_position + 1], atol=1e-6, rtol=1e-6))  # [Codex comment] Confirm all outputs through the completed-block endpoint are unchanged.

    def test_short_prefix_and_early_positions_have_no_nans(self):  # [Codex comment] Cover all-masked global rows and prefixes with zero completed blocks.
        short_output = self.attn(torch.randn(2, 3, 8))  # [Codex comment] Run a prefix shorter than one fixed global block.
        partial_output = self.attn(torch.randn(2, 10, 8))  # [Codex comment] Run a prefix containing completed and partial blocks.
        self.assertFalse(torch.isnan(short_output).any())  # [Codex comment] Confirm skipping the unavailable global branch produces finite outputs.
        self.assertFalse(torch.isnan(partial_output).any())  # [Codex comment] Confirm early all-masked rows are safely zeroed around global softmax.

    def test_average_pooling_projection_initialization(self):  # [Codex comment] Check the requested stable initialization for learned block summaries.
        expected = torch.full_like(self.attn.E, 0.25)  # [Codex comment] Compute one-over-four average weights for each fixed block.
        self.assertTrue(torch.equal(self.attn.E, expected))  # [Codex comment] Confirm compressed keys begin as block averages.
        self.assertTrue(torch.equal(self.attn.F, expected))  # [Codex comment] Confirm compressed values begin as block averages.

    def test_standard_attention_remains_default(self):  # [Codex comment] Protect backward-compatible model construction behavior.
        block = Block()  # [Codex comment] Construct a Transformer block using the unchanged default config value.
        self.assertIsInstance(block.attn, MultiHeadAttention)  # [Codex comment] Confirm standard full causal attention is still selected by default.

    def test_linformer_attention_can_be_selected(self):  # [Codex comment] Verify the Transformer block integrates the new configurable implementation.
        original_attention_type = model_module.attention_type  # [Codex comment] Save the default selector so this test cannot affect later construction.
        try:  # [Codex comment] Ensure the temporary selector change is always restored.
            model_module.attention_type = "causal_blockwise_linformer"  # [Codex comment] Explicitly request the new attention implementation.
            block = model_module.Block().eval()  # [Codex comment] Construct a real Transformer block through the production selector.
            output = block(torch.randn(2, 70, model_module.d_hidden))  # [Codex comment] Exercise a non-divisible generation-like runtime prefix.
            self.assertIsInstance(block.attn, CausalBlockwiseLinformerAttention)  # [Codex comment] Confirm the requested module was selected.
            self.assertEqual(output.shape, (2, 70, model_module.d_hidden))  # [Codex comment] Confirm integration preserves the Transformer block shape.
            self.assertTrue(torch.isfinite(output).all())  # [Codex comment] Confirm integrated local and global branches remain numerically finite.
        finally:  # [Codex comment] Restore shared module state even if an assertion fails.
            model_module.attention_type = original_attention_type  # [Codex comment] Return model construction to standard attention by default.


if __name__ == "__main__":  # [Codex comment] Allow the checks to run directly with python test.py.
    unittest.main()  # [Codex comment] Discover and execute all test methods in this file.
