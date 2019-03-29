from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor


def get_metrics(ref, hypo):
  """
  from SHOW, ATTEND, TELL paper
  ref, dictionary of reference sentences (id, sentence)
  hypo, dictionary of hypothesis sentences (id, sentence)
  score, dictionary of scores
  """
  scorers = [
      (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
      #(Meteor(),"METEOR"),
      #(Rouge(), "ROUGE_L"),
      (Cider(), "CIDEr")
  ]
  final_scores = {}
  for scorer, method in scorers:
      score, scores = scorer.compute_score(ref, hypo)
      if type(score) == list:
          for m, s in zip(method, score):
              final_scores[m] = s
      else:
          final_scores[method] = score
  return final_scores
