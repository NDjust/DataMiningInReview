from django.shortcuts import render
from .SentenceSummary import TextRank
from .SentenceSummary import Sentence


def get_sentence_page(request):
    return render(request, "input_sentence.html")


# Create your views here.
def summarize_sentence(request):
    """ 문장요약 결과물을 보여주는 페이지로 렌더링.

    문장요약 알고리즘을 활용해 TextRank을 만들어 결과물 페에지로 리턴.
    :param request: web resource request
    :return: render html
    """
    full_text = request.GET['fulltext']
    # get TextRank
    textrank1 = TextRank(full_text)
    summary_text = textrank1.summarize(3, verbose=False)  # summary
    print(summary_text)

    # summary_text = "\n".join(summary_text) # list -> String
    return render(request, 'summary.html',
                  {'full': full_text, 'summary': summary_text})  # return summary text
