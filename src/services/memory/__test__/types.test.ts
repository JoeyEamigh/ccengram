import { describe, expect, test } from "bun:test";
import {
  classifyMemorySector,
  isValidSector,
  isValidTier,
  SECTOR_DECAY_RATES,
  ALL_SECTORS,
  type MemorySector,
} from "../types.js";

describe("Memory Sector Classification", () => {
  test("classifies conversation events as episodic", () => {
    const content = "User asked about the authentication flow earlier";
    expect(classifyMemorySector(content)).toBe("episodic");
  });

  test("classifies user requests as episodic", () => {
    const content = "User wanted to refactor the component";
    expect(classifyMemorySector(content)).toBe("episodic");
  });

  test("classifies session references as episodic", () => {
    const content = "In this conversation, we discussed the API design";
    expect(classifyMemorySector(content)).toBe("episodic");
  });

  test("classifies facts as semantic", () => {
    const content = "The auth handler is located in src/auth/handler.ts";
    expect(classifyMemorySector(content)).toBe("semantic");
  });

  test("classifies file locations as semantic", () => {
    const content = "The database module is defined in src/db/index.ts";
    expect(classifyMemorySector(content)).toBe("semantic");
  });

  test("classifies function descriptions as semantic", () => {
    const content = "The function parseConfig has three parameters";
    expect(classifyMemorySector(content)).toBe("semantic");
  });

  test("classifies workflows as procedural", () => {
    const content = "To deploy: first run build, then push to main";
    expect(classifyMemorySector(content)).toBe("procedural");
  });

  test("classifies instructions as procedural", () => {
    const content = "How to set up the development environment";
    expect(classifyMemorySector(content)).toBe("procedural");
  });

  test("classifies commands as procedural", () => {
    const content = "Run bun test to execute the test suite";
    expect(classifyMemorySector(content)).toBe("procedural");
  });

  test("classifies feelings as emotional", () => {
    const content = "Frustrated by the slow test suite";
    expect(classifyMemorySector(content)).toBe("emotional");
  });

  test("classifies preferences as emotional", () => {
    const content = "I prefer using functional components over classes";
    expect(classifyMemorySector(content)).toBe("emotional");
  });

  test("classifies pain points as emotional", () => {
    const content = "This is a major pain point in the workflow";
    expect(classifyMemorySector(content)).toBe("emotional");
  });

  test("classifies insights as reflective", () => {
    const content = "This codebase favors composition over inheritance";
    expect(classifyMemorySector(content)).toBe("reflective");
  });

  test("classifies lessons as reflective", () => {
    const content = "I learned that caching improves performance significantly";
    expect(classifyMemorySector(content)).toBe("reflective");
  });

  test("classifies observations as reflective", () => {
    const content = "I noticed a pattern in the error handling approach";
    expect(classifyMemorySector(content)).toBe("reflective");
  });

  test("defaults to semantic for ambiguous content", () => {
    const content = "The function returns a string";
    expect(classifyMemorySector(content)).toBe("semantic");
  });

  test("defaults to semantic for minimal content", () => {
    const content = "Something";
    expect(classifyMemorySector(content)).toBe("semantic");
  });

  test("handles mixed content by highest match", () => {
    const content = "User asked about the function that is located in src/auth.ts";
    const result = classifyMemorySector(content);
    expect(result === "episodic" || result === "semantic").toBe(true);
  });
});

describe("Sector Decay Rates", () => {
  test("emotional memories decay slowest", () => {
    const rates = Object.entries(SECTOR_DECAY_RATES).sort((a, b) => a[1] - b[1]);
    const slowest = rates[0];
    expect(slowest).toBeDefined();
    if (slowest) {
      expect(slowest[0]).toBe("emotional");
    }
  });

  test("episodic memories decay fastest", () => {
    const rates = Object.entries(SECTOR_DECAY_RATES).sort((a, b) => b[1] - a[1]);
    const fastest = rates[0];
    expect(fastest).toBeDefined();
    if (fastest) {
      expect(fastest[0]).toBe("episodic");
    }
  });

  test("all sectors have decay rates", () => {
    for (const sector of ALL_SECTORS) {
      expect(SECTOR_DECAY_RATES[sector]).toBeGreaterThan(0);
    }
  });

  test("decay rates are in expected order", () => {
    expect(SECTOR_DECAY_RATES.emotional).toBeLessThan(SECTOR_DECAY_RATES.semantic);
    expect(SECTOR_DECAY_RATES.semantic).toBeLessThan(SECTOR_DECAY_RATES.reflective);
    expect(SECTOR_DECAY_RATES.reflective).toBeLessThan(SECTOR_DECAY_RATES.procedural);
    expect(SECTOR_DECAY_RATES.procedural).toBeLessThan(SECTOR_DECAY_RATES.episodic);
  });
});

describe("Type Validators", () => {
  test("isValidSector accepts valid sectors", () => {
    expect(isValidSector("episodic")).toBe(true);
    expect(isValidSector("semantic")).toBe(true);
    expect(isValidSector("procedural")).toBe(true);
    expect(isValidSector("emotional")).toBe(true);
    expect(isValidSector("reflective")).toBe(true);
  });

  test("isValidSector rejects invalid sectors", () => {
    expect(isValidSector("invalid")).toBe(false);
    expect(isValidSector("")).toBe(false);
    expect(isValidSector("EPISODIC")).toBe(false);
  });

  test("isValidTier accepts valid tiers", () => {
    expect(isValidTier("session")).toBe(true);
    expect(isValidTier("project")).toBe(true);
    expect(isValidTier("global")).toBe(true);
  });

  test("isValidTier rejects invalid tiers", () => {
    expect(isValidTier("invalid")).toBe(false);
    expect(isValidTier("")).toBe(false);
    expect(isValidTier("SESSION")).toBe(false);
  });
});

describe("ALL_SECTORS", () => {
  test("contains exactly 5 sectors", () => {
    expect(ALL_SECTORS).toHaveLength(5);
  });

  test("contains all required sectors", () => {
    const expected: MemorySector[] = [
      "episodic",
      "semantic",
      "procedural",
      "emotional",
      "reflective",
    ];
    for (const sector of expected) {
      expect(ALL_SECTORS).toContain(sector);
    }
  });
});
