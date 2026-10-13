"""Shared educational prose for reports, independent of arithmetic and evidence.

Readers should not need to inspect the implementation to interpret its results.
Introduce terminology before using it, connect limitations to their practical
meaning, and distinguish a worked example from a claim about a recorded run.
These functions return fixed, trusted HTML; labels from experiment data are still
escaped by the report renderers. No counters, models or source JSON are changed.
"""


def reading_guide():
    """Introduce the report's purpose and the three kinds of numbers it contains."""
    return (
        '<section class="panel"><h2>A starting point for reading this report</h2>'
        '<p>We want to understand what a matrix-multiplication algorithm actually asks the computer to do. '
        'An algorithm may save multiplications by doing more additions, preparing complicated combinations '
        'of entries, or storing extra temporary matrices. The graphs help us explore those trade-offs. '
        'They are not a race to see which Python function finishes first.</p>'
        '<p>There are three different kinds of evidence here. <strong>Measured counts</strong> come from '
        'running our code and counting its arithmetic calls. <strong>Predictions and projections</strong> '
        'come from calculating how the recipe should behave; a projected large matrix was not necessarily '
        'executed. <strong>Theoretical exponents</strong> describe long-term growth under stated assumptions. '
        'A small successful experiment helps check the implementation, but cannot establish that long-term '
        'growth on its own.</p>'
        '<p>For example, Strassen can multiply two 2×2 blocks using seven smaller products instead of the '
        'eight used by the ordinary row-by-column method. It pays for that saving with extra sums and '
        'differences. Repeating the same idea on larger blocks is called recursion: a smaller multiplication '
        'uses the same recipe again until it reaches a block we multiply directly.</p>'
        '<p>Our authoritative calculations use exact integers and fractions, or exact real/imaginary pairs '
        'when a recipe needs complex numbers. Floating-point experiments are separately labelled accuracy '
        'diagnostics. Rounded graph labels, logarithms and elapsed-time readings do not replace the exact '
        'arithmetic used to check a recipe.</p></section>')


def terminology_guide():
    """Provide a small glossary with definitions tied to the program's reports."""
    return (
        '<details class="panel"><summary>A small glossary: blocks, stages, ranks and other terms</summary>'
        '<dl><dt><strong>Matrix size and shape</strong></dt><dd>A square matrix of size n has n rows and n columns. '
        'A multiplication shape (m,k,n) means an m×k matrix multiplied by a k×n matrix, producing an m×n result. '
        'For example, (2,3,4) takes a 2×3 and a 3×4 matrix and produces a 2×4 matrix.</dd>'
        '<dt><strong>Block</strong></dt><dd>A smaller rectangular piece of a matrix. Splitting a matrix in half '
        'along both rows and columns gives four blocks. A multiplication of two blocks is itself a matrix '
        'multiplication, so we may apply another recipe to it.</dd>'
        '<dt><strong>Stage and recursion depth</strong></dt><dd>A stage is the recipe used at one level of '
        'the calculation. Depth zero is the original request; depth one is its smaller child problems; '
        'depth two is their children. A repeating schedule chooses which recipe each level uses.</dd>'
        '<dt><strong>Radix and rank</strong></dt><dd>For a square stage, its radix tells us how much each '
        'matrix dimension shrinks in a child problem. Radix two means half-sized children. Its rank here '
        'counts the child products in the available recipe: Strassen has seven. That is seven block '
        'multiplications, not necessarily seven individual-number multiplications. The word rank does not '
        'claim that our available recipe is the smallest possible decomposition. Rectangular stages have '
        'separate row, inner and column subdivision factors.</dd>'
        '<dt><strong>Leaf and cutoff</strong></dt><dd>A leaf is a problem we stop splitting and multiply '
        'by the direct row-by-column method. The cutoff is our chosen maximum leaf dimension. It changes '
        'the balance between saved products and extra combination work.</dd>'
        '<dt><strong>Data product and coefficient product</strong></dt><dd>A data product multiplies values '
        'that depend on the input matrices. A coefficient product multiplies a value by a fixed number '
        'in the recipe, such as 1/2. Both involve work, but only the first kind is used when reporting '
        'the recipe’s bilinear product count.</dd>'
        '<dt><strong>Field step and atomic step</strong></dt><dd>A field step acts on one mathematical '
        'number. An atomic step is one of the smaller scalar calls our code exposes. For example, our '
        'exact complex multiplication is one field multiplication implemented with four real multiplications '
        'and two real sums or differences. Those totals describe the same work at different levels, '
        'so adding them together would count it twice.</dd>'
        '<dt><strong>Exponent and omega (ω)</strong></dt><dd>An algorithm exponent p describes how its '
        'arithmetic grows, roughly like n raised to p for large square matrices. An exponent of three '
        'means that doubling n gives about eight times the leading work. Omega is the infimum of exponents '
        'achievable by matrix algorithms in the stated arithmetic setting; it is not the exponent of every '
        'algorithm in this report. Small-range fitted slopes can differ from the theoretical exponent. '
        'The expression log₂(7) means “the power p for which 2 raised to p is 7”, approximately 2.807. '
        'It captures Strassen’s sevenfold increase in descendant products when the dimensions double.</dd>'
        '<dt><strong>Recurrence</strong></dt><dd>A rule describing a problem in terms of its smaller '
        'children. In T(n)=7 T(n/2)+O(n²), T(n) denotes the work for size n: seven half-size products '
        'plus combination work whose growth is bounded by a constant times n². The rule must be justified '
        'by the construction; it is not guessed from a graph.</dd>'
        '<dt><strong>Setup, verification and measurement overhead</strong></dt><dd>Setup prepares or '
        'searches for a recipe before applying it to matrix data. Verification checks its answer or '
        'algebraic identity. Measurement overhead is the extra work our counters and allocation probes '
        'perform. Their costs are reported with their own boundaries so that they are not silently '
        'mistaken for arithmetic needed by the multiplication itself.</dd></dl></details>')


def composition_guide():
    """Explain a repeating stage cycle with a concrete two-level example."""
    return (
        '<p>A composition is a plan for choosing the recipe at each level of recursion. Suppose the plan '
        'says “Strassen, then Strassen–Winograd”. We use Strassen to break the original problem into smaller '
        'products. Each of those products uses Strassen–Winograd at the next level. If further splitting '
        'is needed, the plan starts again with Strassen and continues in the same order.</p>'
        '<p>Both of those stages halve the dimensions and request seven child products. After two levels, '
        'each descendant product is one quarter of the original size, and there are 7×7=49 descendant '
        'products. This is why the table combines stage subdivision factors and product counts by '
        'multiplication: two halvings give a total size reduction of 2×2=4, and two sets of seven give '
        'a total of 49. These are block products; the eventual scalar cost also depends on where we stop '
        'splitting and on the additions between stages.</p>'
        '<p>The table lists the schedules actually used in this report. A rectangular schedule may shrink '
        'the three matrix dimensions by different factors at an individual stage. Its full cycle must '
        'be balanced before we describe it with one square base size.</p>'
        '<p>Combining recipes gives us a program we can run and inspect. It does not automatically give a '
        'better exponent than the original methods: this particular two-stage example still has '
        'log₄(49)=log₂(7). A claim of mathematical improvement needs a verified construction with a '
        'strictly smaller justified exponent, not simply a new arrangement of familiar recipes.</p>')


def construction_guide():
    """Introduce coefficient-search and CW pages before their specialist results."""
    return (
        '<h2>What this experiment is trying to build</h2>'
        '<p>A multiplication recipe specifies which input entries to add or scale, which resulting '
        'combinations to multiply, and how to assemble the answers. The fixed numbers in those '
        'instructions are called coefficients. Here we build, transform or search for such recipes '
        'using exact arithmetic, then check that the resulting algebra agrees with the ordinary '
        'matrix-multiplication rule.</p>'
        '<p>Some pages investigate the Coppersmith–Winograd structure, abbreviated CW. It is a mathematical '
        'building block used in research on fast multiplication. Its tensor describes a relationship '
        'between three groups of coordinates. A tensor is not automatically one ordinary matrix '
        'multiplication: it may describe several separate products or a relationship that needs '
        'further work before it can be used as a matrix recipe.</p>'
        '<p>A complete coefficient check verifies the proposed algebraic identity, rather than trying '
        'only a few input matrices. We also execute small examples to check the program’s outputs and '
        'count its work. These are complementary checks: a correct mathematical identity does not '
        'prove every detail of the Python interpreter or every possible execution path.</p>')
