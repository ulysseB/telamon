{{#each arguments~}}
    {{#if ../arg_names}}{{lookup ../arg_names @index}}{{else}}{{this.[0]}}{{/if}},
{{~/each}}
