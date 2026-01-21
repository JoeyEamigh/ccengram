import { describe, expect, test } from 'bun:test';
import {
  ValidationError,
  validateArray,
  validateEnum,
  validateNumber,
  validateOptionalArray,
  validateOptionalEnum,
  validateOptionalNumber,
  validateOptionalString,
  validateString,
} from '../validate.js';

describe('validateString', () => {
  test('returns valid string', () => {
    expect(validateString('hello', 'field')).toBe('hello');
  });

  test('throws for non-string', () => {
    expect(() => validateString(123, 'field')).toThrow(ValidationError);
    expect(() => validateString(null, 'field')).toThrow(ValidationError);
    expect(() => validateString(undefined, 'field')).toThrow(ValidationError);
    expect(() => validateString({}, 'field')).toThrow(ValidationError);
  });

  test('throws for empty string by default', () => {
    expect(() => validateString('', 'field')).toThrow(ValidationError);
  });

  test('allows empty string when allowEmpty is true', () => {
    expect(validateString('', 'field', { allowEmpty: true })).toBe('');
  });

  test('enforces minLength', () => {
    expect(() => validateString('ab', 'field', { minLength: 3 })).toThrow(ValidationError);
    expect(validateString('abc', 'field', { minLength: 3 })).toBe('abc');
  });

  test('enforces maxLength', () => {
    expect(() => validateString('abcd', 'field', { maxLength: 3 })).toThrow(ValidationError);
    expect(validateString('abc', 'field', { maxLength: 3 })).toBe('abc');
  });

  test('error contains field name', () => {
    try {
      validateString(123, 'myField');
    } catch (e) {
      expect(e).toBeInstanceOf(ValidationError);
      expect((e as ValidationError).field).toBe('myField');
      expect((e as ValidationError).message).toContain('myField');
    }
  });
});

describe('validateNumber', () => {
  test('returns valid number', () => {
    expect(validateNumber(42, 'field')).toBe(42);
    expect(validateNumber(0, 'field')).toBe(0);
    expect(validateNumber(-5, 'field')).toBe(-5);
    expect(validateNumber(3.14, 'field')).toBe(3.14);
  });

  test('throws for non-number', () => {
    expect(() => validateNumber('42', 'field')).toThrow(ValidationError);
    expect(() => validateNumber(null, 'field')).toThrow(ValidationError);
    expect(() => validateNumber(undefined, 'field')).toThrow(ValidationError);
    expect(() => validateNumber({}, 'field')).toThrow(ValidationError);
  });

  test('throws for NaN by default', () => {
    expect(() => validateNumber(NaN, 'field')).toThrow(ValidationError);
  });

  test('allows NaN when allowNaN is true', () => {
    expect(validateNumber(NaN, 'field', { allowNaN: true })).toBeNaN();
  });

  test('enforces min', () => {
    expect(() => validateNumber(5, 'field', { min: 10 })).toThrow(ValidationError);
    expect(validateNumber(10, 'field', { min: 10 })).toBe(10);
  });

  test('enforces max', () => {
    expect(() => validateNumber(15, 'field', { max: 10 })).toThrow(ValidationError);
    expect(validateNumber(10, 'field', { max: 10 })).toBe(10);
  });

  test('enforces both min and max', () => {
    expect(() => validateNumber(0, 'field', { min: 1, max: 10 })).toThrow(ValidationError);
    expect(() => validateNumber(11, 'field', { min: 1, max: 10 })).toThrow(ValidationError);
    expect(validateNumber(5, 'field', { min: 1, max: 10 })).toBe(5);
  });
});

describe('validateEnum', () => {
  const COLORS = ['red', 'green', 'blue'] as const;

  test('returns valid enum value', () => {
    expect(validateEnum('red', 'field', COLORS)).toBe('red');
    expect(validateEnum('green', 'field', COLORS)).toBe('green');
    expect(validateEnum('blue', 'field', COLORS)).toBe('blue');
  });

  test('throws for invalid value', () => {
    expect(() => validateEnum('yellow', 'field', COLORS)).toThrow(ValidationError);
  });

  test('throws for non-string', () => {
    expect(() => validateEnum(123, 'field', COLORS)).toThrow(ValidationError);
    expect(() => validateEnum(null, 'field', COLORS)).toThrow(ValidationError);
  });

  test('error message lists allowed values', () => {
    try {
      validateEnum('yellow', 'color', COLORS);
    } catch (e) {
      expect((e as ValidationError).message).toContain('red');
      expect((e as ValidationError).message).toContain('green');
      expect((e as ValidationError).message).toContain('blue');
    }
  });
});

describe('validateOptionalString', () => {
  test('returns undefined for undefined', () => {
    expect(validateOptionalString(undefined, 'field')).toBeUndefined();
  });

  test('returns undefined for null', () => {
    expect(validateOptionalString(null, 'field')).toBeUndefined();
  });

  test('validates when value is present', () => {
    expect(validateOptionalString('hello', 'field')).toBe('hello');
    expect(() => validateOptionalString(123, 'field')).toThrow(ValidationError);
  });

  test('passes options through', () => {
    expect(() => validateOptionalString('ab', 'field', { minLength: 3 })).toThrow(ValidationError);
  });
});

describe('validateOptionalNumber', () => {
  test('returns undefined for undefined', () => {
    expect(validateOptionalNumber(undefined, 'field')).toBeUndefined();
  });

  test('returns undefined for null', () => {
    expect(validateOptionalNumber(null, 'field')).toBeUndefined();
  });

  test('validates when value is present', () => {
    expect(validateOptionalNumber(42, 'field')).toBe(42);
    expect(() => validateOptionalNumber('42', 'field')).toThrow(ValidationError);
  });

  test('passes options through', () => {
    expect(() => validateOptionalNumber(5, 'field', { min: 10 })).toThrow(ValidationError);
  });
});

describe('validateOptionalEnum', () => {
  const SIZES = ['small', 'medium', 'large'] as const;

  test('returns undefined for undefined', () => {
    expect(validateOptionalEnum(undefined, 'field', SIZES)).toBeUndefined();
  });

  test('returns undefined for null', () => {
    expect(validateOptionalEnum(null, 'field', SIZES)).toBeUndefined();
  });

  test('validates when value is present', () => {
    expect(validateOptionalEnum('small', 'field', SIZES)).toBe('small');
    expect(() => validateOptionalEnum('xl', 'field', SIZES)).toThrow(ValidationError);
  });
});

describe('validateArray', () => {
  test('returns validated array', () => {
    const result = validateArray(['a', 'b', 'c'], 'field', (item) => {
      if (typeof item !== 'string') throw new Error('not string');
      return item.toUpperCase();
    });
    expect(result).toEqual(['A', 'B', 'C']);
  });

  test('throws for non-array', () => {
    expect(() => validateArray('not array', 'field', (x) => x)).toThrow(ValidationError);
    expect(() => validateArray({}, 'field', (x) => x)).toThrow(ValidationError);
    expect(() => validateArray(null, 'field', (x) => x)).toThrow(ValidationError);
  });

  test('item validator receives index', () => {
    const indices: number[] = [];
    validateArray(['a', 'b', 'c'], 'field', (item, index) => {
      indices.push(index);
      return item;
    });
    expect(indices).toEqual([0, 1, 2]);
  });

  test('propagates item validation errors', () => {
    expect(() =>
      validateArray([1, 'two', 3], 'field', (item) => {
        if (typeof item !== 'number') throw new ValidationError('not number', 'item');
        return item;
      }),
    ).toThrow(ValidationError);
  });
});

describe('validateOptionalArray', () => {
  test('returns undefined for undefined', () => {
    expect(validateOptionalArray(undefined, 'field', (x) => x)).toBeUndefined();
  });

  test('returns undefined for null', () => {
    expect(validateOptionalArray(null, 'field', (x) => x)).toBeUndefined();
  });

  test('validates when value is present', () => {
    const result = validateOptionalArray([1, 2, 3], 'field', (item) => item);
    expect(result).toEqual([1, 2, 3]);
  });
});
