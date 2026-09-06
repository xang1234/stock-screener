import { readFileSync } from 'node:fs';

import Ajv2020 from 'ajv/dist/2020.js';
import addFormats from 'ajv-formats';
import standaloneCode from 'ajv/dist/standalone/index.js';

const VIRTUAL_ID = 'virtual:options-validators';
const RESOLVED_VIRTUAL_ID = `\0${VIRTUAL_ID}`;
const SCHEMA_PATH = new URL('../src/features/options/optionsSchema.json', import.meta.url);

const replaceRuntimeImports = (generatedCode) => {
  let needsUcs2Length = false;
  let needsFormats = false;

  const withUcs2Length = generatedCode.replace(
    /const (\w+) = require\("ajv\/dist\/runtime\/ucs2length"\)\.default;/g,
    (_match, binding) => {
      needsUcs2Length = true;
      return `const ${binding} = ucs2length;`;
    },
  );
  const withFormats = withUcs2Length.replace(
    /const (\w+) = require\("ajv-formats\/dist\/formats"\)\.fullFormats(?:\.([\w-]+)|\["([^"]+)"\]);/g,
    (_match, binding, dottedName, bracketedName) => {
      needsFormats = true;
      return `const ${binding} = fullFormats[${JSON.stringify(dottedName || bracketedName)}];`;
    },
  );

  if (withFormats.includes('require(')) {
    throw new Error('Options validator generation produced an unsupported runtime import');
  }

  const imports = [
    needsUcs2Length
      ? 'import ucs2LengthModule from "ajv/dist/runtime/ucs2length.js";\nconst ucs2length = ucs2LengthModule.default ?? ucs2LengthModule;'
      : '',
    needsFormats
      ? 'import formatsModule from "ajv-formats/dist/formats.js";\nconst fullFormats = formatsModule.fullFormats ?? formatsModule.default?.fullFormats;'
      : '',
  ].filter(Boolean).join('\n');

  return `${imports}\n${withFormats}`;
};

const buildValidatorModule = () => {
  const wireSchema = JSON.parse(readFileSync(SCHEMA_PATH, 'utf8'));
  const ajv = new Ajv2020({
    allErrors: true,
    strict: true,
    code: { esm: true, source: true },
  });
  addFormats(ajv);
  ajv.addSchema(wireSchema.models.manifest, 'manifest');
  ajv.addSchema(wireSchema.models.command_center, 'commandCenter');
  ajv.addSchema(wireSchema.models.symbol_detail, 'symbolDetail');

  return replaceRuntimeImports(standaloneCode(ajv, {
    manifest: 'manifest',
    commandCenter: 'commandCenter',
    symbolDetail: 'symbolDetail',
  }));
};

export const optionsValidatorsPlugin = () => ({
  name: 'options-validators',
  resolveId(id) {
    return id === VIRTUAL_ID ? RESOLVED_VIRTUAL_ID : null;
  },
  load(id) {
    return id === RESOLVED_VIRTUAL_ID ? buildValidatorModule() : null;
  },
});
