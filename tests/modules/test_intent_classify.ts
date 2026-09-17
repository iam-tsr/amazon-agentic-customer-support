import { classifyIntent } from '../../src/intent_classify';

const cases = [
    { input: 'Can you tell me the status of my order?', expected: 'question'},
    { input: 'I love this product! It works perfectly.', expected: 'positive'},
    { input: 'This is just a random statement.', expected: 'other'},
];


let passed = 0;
for (const { input, expected } of cases) {
    const result = await classifyIntent(input);
    const ok = result === expected;
    if (ok) passed++;

    console.log(
        `${ok ? "OK " : "X  "} want=${expected.padEnd(12)} got=${result}`.padEnd(46) +
        ` ${input.slice(0, 64)}`
    );
}

console.log(`\n${passed}/${cases.length} passed`);