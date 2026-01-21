export class ValidationError extends Error {
  constructor(
    message: string,
    public readonly field: string,
  ) {
    super(message);
    this.name = 'ValidationError';
  }
}

export type StringOptions = {
  maxLength?: number;
  minLength?: number;
  allowEmpty?: boolean;
};

export type NumberOptions = {
  min?: number;
  max?: number;
  allowNaN?: boolean;
};

export function validateString(value: unknown, name: string, opts: StringOptions = {}): string {
  if (typeof value !== 'string') {
    throw new ValidationError(`${name} must be a string`, name);
  }

  if (!opts.allowEmpty && value.length === 0) {
    throw new ValidationError(`${name} cannot be empty`, name);
  }

  if (opts.minLength !== undefined && value.length < opts.minLength) {
    throw new ValidationError(`${name} must be at least ${opts.minLength} characters`, name);
  }

  if (opts.maxLength !== undefined && value.length > opts.maxLength) {
    throw new ValidationError(`${name} must be at most ${opts.maxLength} characters`, name);
  }

  return value;
}

export function validateNumber(value: unknown, name: string, opts: NumberOptions = {}): number {
  if (typeof value !== 'number') {
    throw new ValidationError(`${name} must be a number`, name);
  }

  if (!opts.allowNaN && Number.isNaN(value)) {
    throw new ValidationError(`${name} cannot be NaN`, name);
  }

  if (opts.min !== undefined && value < opts.min) {
    throw new ValidationError(`${name} must be at least ${opts.min}`, name);
  }

  if (opts.max !== undefined && value > opts.max) {
    throw new ValidationError(`${name} must be at most ${opts.max}`, name);
  }

  return value;
}

export function validateEnum<T extends string>(
  value: unknown,
  name: string,
  allowed: readonly T[],
): T {
  if (typeof value !== 'string') {
    throw new ValidationError(`${name} must be a string`, name);
  }

  if (!allowed.includes(value as T)) {
    throw new ValidationError(`${name} must be one of: ${allowed.join(', ')}`, name);
  }

  return value as T;
}

export function validateOptionalString(
  value: unknown,
  name: string,
  opts: StringOptions = {},
): string | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  return validateString(value, name, opts);
}

export function validateOptionalNumber(
  value: unknown,
  name: string,
  opts: NumberOptions = {},
): number | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  return validateNumber(value, name, opts);
}

export function validateOptionalEnum<T extends string>(
  value: unknown,
  name: string,
  allowed: readonly T[],
): T | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  return validateEnum(value, name, allowed);
}

export function validateArray<T>(
  value: unknown,
  name: string,
  itemValidator: (item: unknown, index: number) => T,
): T[] {
  if (!Array.isArray(value)) {
    throw new ValidationError(`${name} must be an array`, name);
  }

  return value.map((item, index) => itemValidator(item, index));
}

export function validateOptionalArray<T>(
  value: unknown,
  name: string,
  itemValidator: (item: unknown, index: number) => T,
): T[] | undefined {
  if (value === undefined || value === null) {
    return undefined;
  }
  return validateArray(value, name, itemValidator);
}
