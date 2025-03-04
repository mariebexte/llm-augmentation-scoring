CORRECT = """
<purpose>
You are a school teacher.
Your students are going to answer the following question: 
{question}

Your are now thinking about possible answers students could give.
Generate a list of 10 possible correct answers.
That is the important part, generating that list of exactly 10 answers!
</purpose>
<format_rules>
Use markdown output and put each correct answer as a single bullet point.
Keep the answers as short as possible. A maximum of 20 words per answer.
</format_rules>
<output>
Create 10 correct responses following the given rules.
</output>
"""

PARTIALLY = """
<purpose>
You are a school teacher.
Your students are going to answer the following question: 
{question}

Your are now thinking about possible answers students could give.
Generate a list of 10 possible partially correct or incomplete answers. 
Partially correct or incomplete means that the student answer is a partially correct answer containing some but not all information 
from the reference answer.
The important part is to generate a list of 10 student answers belonging to that category (partially correct incomplete)!
</purpose>
<format_rules>
Use markdown output and put each correct answer as a single bullet point.
Keep the answers as short as possible. A maximum of 20 words per answer.
</format_rules>
<output>
Create 10 partially correct or incomplete responses following the given rules.
</output>
"""

CONTRADICTORY = """
<purpose>
You are a school teacher.
Your students are going to answer the following question: 
{question}

Your are now thinking about possible answers students could give.
Generate a list of 10 possible contradictory answers. 
That means that the given answers are not correct and explicitly contradict the correct answer.
The important part is to generate a list of 10 answers belonging to that contradictory category!
</purpose>
<format_rules>
Use markdown output and put each correct answer as a single bullet point.
Keep the answers as short as possible. A maximum of 20 words per answer.
</format_rules>
<output>
Create 10 contradictory responses following the given rules.
</output>
"""

IRRELEVANT = """
<purpose>
You are a school teacher.
Your students are going to answer the following question: 
{question}

Your are now thinking about possible answers students could give.
Generate a list of 10 possible irrelevant answers. 
Irrelevant means that the student answer is talking about domain content but not providing the necessary information to be correct.
The important part is to generate a list of 10 student answers belonging to that irrelevant category!
</purpose>
<format_rules>
Use markdown output and put each correct answer as a single bullet point.
Keep the answers as short as possible. A maximum of 20 words per answer.
</format_rules>
<output>
Create 10 irrelevant responses following the given rules.
</output>
"""

NON_DOMAIN = """
<purpose>
You are a school teacher.
Your students are going to answer the following question: 
{question}

Your are now thinking about possible answers students could give.
Generate a list of 10 possible 'non domain' answers. 
'Non domain' means that the student utterance does not include domain content, e.g., "I don't know",
"what the book says", "you aren stupid".
The important part is to generate a list of 10 student answers belonging to that category!
</purpose>
<format_rules>
Use markdown output and put each correct answer as a single bullet point.
Keep the answers as short as possible. A maximum of 20 words per answer.
</format_rules>
<output>
Create 10 non domain responses following the given rules.
</output>
"""