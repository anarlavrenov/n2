# N2

N2 є моделлю Automatic Speech Recognition (або Speech-To-Text) англійською мовою. Написана з нуля на PyTorch. Розроблена [Анаром Лавреновим, Head of AI компанії SPUNCH](https://www.linkedin.com/mynetwork/). N2 має на меті заохотити вітчизняних Дата Сайнтистів до розвитку українскього напряму ШІ. 

### Пайплайн тренування моделі
![image](https://github.com/anarlavrenov/n2/blob/main/pipeline_diagram.webp)

### Графік лоссу протягом тренування
![image](https://github.com/anarlavrenov/n2/blob/main/loss.png)


Модель навчена на датасеті LibriSpeech-clean-100 з кастомною аугментацією данних. Валідацію проходила на іншому датасеті: LJSpeech.
Декілька прикладів роботи моделі на LJSpeech:

```
pred: studie's indicate that there is some utility ind attempting to desg nate certain buildings as in volving a higher risk of an others
true: the studies indicate that there is some utility in attempting to designate certain buildings as involving a higher risk than others

pred: ad coordination might be achieved to a greater extet than seems now to be contemplated without intefearence but the primary mession of ecagent se involved
true: that coordination might be achieved to a greater extent than seems now to be contemplated without interference with the primary mission of each agency involved

pred: at rickon instructions might come into the hands of local newspapers to the prejidice of the procautions described
true: that written instructions might come into the hands of local newspapers to the prejudice of the precautions described

pred: and requests for inditional personal were not made because of the studies than being conducted
true: and requests for additional personnel were not made because of the studies then being conducted

pred: caselon of e cef be i agent average twenty to twenty five and he felt that this was high
true: the caseload of each fbi agent averaged  to  and he felt that this was high

```
