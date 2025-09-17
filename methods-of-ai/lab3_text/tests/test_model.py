import torch

from lab3_text.train_spam import HybridBagModel


def test_hybrid_bag_model_outputs_logits_per_sample():
    model = HybridBagModel(word_vocab=100, char_vocab=50)
    word_tokens = torch.randint(0, 100, (40,))
    char_tokens = torch.randint(0, 50, (80,))
    word_offsets = torch.tensor([0, 10, 25, 35])
    char_offsets = torch.tensor([0, 20, 45, 70])
    logits = model(word_tokens, word_offsets, char_tokens, char_offsets)
    assert logits.shape == (4,)
