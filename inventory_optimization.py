import pandas as pd
import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD, LpStatus, value

# 1. ЗАГРУЖАЕМ ДАННЫЕ
df = df = pd.read_csv(r'C:\Users\Dell\Downloads\archive(3)\supply_chain_data.csv')

print("Первые 5 строк:")
print(df.head())

print("\nРазмер таблицы:")
print(df.shape)



# 2. БАЗОВАЯ ИНФОРМАЦИЯ О ДАННЫХ
columns_for_stats = [
    'Price',
    'Stock levels',
    'Lead times',
    'Order quantities',
    'Revenue generated',
    'Manufacturing costs',
    'Costs'
]

print("\nСтатистика по числовым колонкам:")
print(df[columns_for_stats].describe().round(2))

print("\nКоличество товаров по типам:")
print(df['Product type'].value_counts())

total_revenue = df['Revenue generated'].sum()
total_costs = df['Costs'].sum()
total_profit = total_revenue - total_costs

print("\nОбщие финансовые показатели:")
print(f"Общая выручка: ${total_revenue:,.0f}")
print(f"Общие затраты: ${total_costs:,.0f}")
print(f"Общая прибыль: ${total_profit:,.0f}")



# 3. ABC АНАЛИЗ
df_abc = df[['SKU', 'Product type', 'Revenue generated', 'Stock levels', 'Costs']].copy()

df_abc = df_abc.sort_values(by='Revenue generated', ascending=False)

df_abc['Cumulative Revenue'] = df_abc['Revenue generated'].cumsum()
df_abc['Cumulative %'] = df_abc['Cumulative Revenue'] / total_revenue * 100


def get_abc_group(percent):
    if percent <= 80:
        return 'A'
    elif percent <= 95:
        return 'B'
    else:
        return 'C'


df_abc['ABC'] = df_abc['Cumulative %'].apply(get_abc_group)

print("\nABC анализ , первые 10 товаров:")
print(df_abc[['SKU', 'Product type', 'Revenue generated', 'Cumulative %', 'ABC']].head(10).round(2))

print("\nСводка по ABC группам:")
for group in ['A', 'B', 'C']:
    group_data = df_abc[df_abc['ABC'] == group]

    count_products = len(group_data)
    group_revenue = group_data['Revenue generated'].sum()
    average_stock = group_data['Stock levels'].mean()

    print(f"\nГруппа {group}:")
    print(f"Количество товаров: {count_products}")
    print(f"Общая выручка: ${group_revenue:,.0f}")
    print(f"Средний остаток: {average_stock:.2f}")



# 4. XYZ АНАЛИЗ
# В нормальном XYZ анализе нужен спрос по месяцам/дням.
# В этом датасете такого нет, поэтому делаем упрощенный вариант:
# сравниваем Order quantities каждого товара со средним значением.
xyz_data = df[['SKU', 'Product type', 'Order quantities', 'Number of products sold']].copy()

average_order_quantity = df['Order quantities'].mean()

xyz_data['CV'] = abs(xyz_data['Order quantities'] - average_order_quantity) / average_order_quantity


def get_xyz_group(cv):
    if cv <= 0.25:
        return 'X'
    elif cv <= 0.50:
        return 'Y'
    else:
        return 'Z'


xyz_data['XYZ'] = xyz_data['CV'].apply(get_xyz_group)

print("\nXYZ анализ , количество товаров по группам:")
print(xyz_data['XYZ'].value_counts())

xyz_data = xyz_data.merge(df_abc[['SKU', 'ABC']], on='SKU')
xyz_data['ABC_XYZ'] = xyz_data['ABC'] + xyz_data['XYZ']

print("\nМатрица ABC-XYZ:")
print(pd.crosstab(xyz_data['ABC'], xyz_data['XYZ']))



# 5. ПРОВЕРКА STOCKOUT
stockout = df[df['Stock levels'] == 0].copy()

stockout = stockout[['SKU', 'Product type', 'Revenue generated', 'Lead times']]
stockout = stockout.merge(xyz_data[['SKU', 'ABC_XYZ']], on='SKU')
stockout = stockout.sort_values(by='Revenue generated', ascending=False)

print("\nТовары, которых нет на складе:")
print(stockout)

print(f"\nВсего товаров в stockout: {len(stockout)}")
print(f"Потенциально потерянная выручка: ${stockout['Revenue generated'].sum():,.0f}")



# 6. EOQ
ORDERING_COST = 50
HOLDING_RATE = 0.20

eoq_data = df[['SKU', 'Product type', 'Price', 'Number of products sold',
               'Stock levels', 'Lead times']].copy()

eoq_data['D'] = eoq_data['Number of products sold']
eoq_data['S'] = ORDERING_COST
eoq_data['H'] = eoq_data['Price'] * HOLDING_RATE

eoq_data['EOQ'] = np.sqrt((2 * eoq_data['D'] * eoq_data['S']) / eoq_data['H'])
eoq_data['EOQ'] = eoq_data['EOQ'].round(0)

eoq_data = eoq_data.merge(xyz_data[['SKU', 'ABC_XYZ']], on='SKU')
eoq_data = eoq_data.merge(df_abc[['SKU', 'ABC']], on='SKU')

print("\nEOQ , первые 10 товаров:")
print(eoq_data[['SKU', 'Product type', 'Price', 'D', 'EOQ', 'Stock levels', 'ABC_XYZ']].head(10).round(2))

print("\nСредний EOQ по ABC группам:")
for group in ['A', 'B', 'C']:
    group_eoq = eoq_data[eoq_data['ABC'] == group]['EOQ'].mean()
    print(f"Группа {group}: {group_eoq:.0f}")


# Проверяем, где текущий запас меньше EOQ
low_stock_eoq = eoq_data[eoq_data['Stock levels'] < eoq_data['EOQ']].copy()

low_stock_eoq['Deficit'] = low_stock_eoq['EOQ'] - low_stock_eoq['Stock levels']
low_stock_eoq['Deficit'] = low_stock_eoq['Deficit'].round(0)

low_stock_eoq = low_stock_eoq.sort_values(by='Deficit', ascending=False)

print("\nТовары, где текущий запас ниже EOQ:")
print(low_stock_eoq[['SKU', 'Stock levels', 'EOQ', 'Deficit', 'ABC_XYZ']].head(10))

print(f"\nВсего товаров ниже EOQ: {len(low_stock_eoq)}")



# 7. SAFETY STOCK И ROP
Z = 1.65  # примерно 95% service level

eoq_data['Daily demand'] = eoq_data['D'] / 365

# Упрощенное предположение:
# разброс спроса = 20% от дневного спроса
eoq_data['Sigma'] = eoq_data['Daily demand'] * 0.20

eoq_data['Safety Stock'] = Z * eoq_data['Sigma'] * np.sqrt(eoq_data['Lead times'])
eoq_data['Safety Stock'] = eoq_data['Safety Stock'].round(0)

eoq_data['ROP'] = eoq_data['Daily demand'] * eoq_data['Lead times'] + eoq_data['Safety Stock']
eoq_data['ROP'] = eoq_data['ROP'].round(0)

print("\nSafety Stock и ROP , первые 10 товаров:")
print(eoq_data[['SKU', 'ABC_XYZ', 'Stock levels', 'EOQ', 'Safety Stock', 'ROP']].head(10))


# Находим товары, которые нужно заказать сейчас
reorder_now = eoq_data[eoq_data['Stock levels'] <= eoq_data['ROP']].copy()

reorder_now = reorder_now[['SKU', 'ABC', 'ABC_XYZ', 'Stock levels', 'ROP', 'EOQ']]
reorder_now = reorder_now.sort_values(by='ABC')

print("\nТовары ниже ROP , нужно заказывать:")
print(reorder_now)

print(f"\nВсего товаров для заказа: {len(reorder_now)}")

critical_items = reorder_now[reorder_now['ABC'] == 'A']

print("\nКритические товары из группы A:")
print(critical_items)

print(f"\nКоличество критических товаров: {len(critical_items)}")



# 8. ПРОСТАЯ LP ОПТИМИЗАЦИЯ
# Ограничения:
# 1. Для товаров A: текущий запас + заказ >= ROP
# 2. Заказ не должен быть больше EOQ

skus = list(eoq_data['SKU'])

price = dict(zip(eoq_data['SKU'], eoq_data['Price']))
current_stock = dict(zip(eoq_data['SKU'], eoq_data['Stock levels']))
rop = dict(zip(eoq_data['SKU'], eoq_data['ROP']))
eoq = dict(zip(eoq_data['SKU'], eoq_data['EOQ']))
abc = dict(zip(eoq_data['SKU'], eoq_data['ABC']))
abc_xyz = dict(zip(eoq_data['SKU'], eoq_data['ABC_XYZ']))

problem = LpProblem("Inventory_Optimization", LpMinimize)

order_qty = {}

for sku in skus:
    order_qty[sku] = LpVariable(f"order_{sku}", lowBound=0)

# Целевая функция: минимизировать стоимость заказа
problem += lpSum(price[sku] * order_qty[sku] for sku in skus)

# Ограничение для товаров группы A
for sku in skus:
    if abc[sku] == 'A':
        problem += current_stock[sku] + order_qty[sku] >= rop[sku]

# Ограничение: не заказывать больше EOQ
for sku in skus:
    problem += order_qty[sku] <= eoq[sku]

problem.solve(PULP_CBC_CMD(msg=0))

print("\nLP оптимизация:")
print(f"Статус решения: {LpStatus[problem.status]}")
print(f"Минимальная стоимость пополнения группы A: ${value(problem.objective):,.2f}")




orders = []

for sku in skus:
    qty = value(order_qty[sku])

    if qty is not None and qty > 0.5:
        order_cost = price[sku] * qty
        final_stock = current_stock[sku] + qty

        orders.append({
            'SKU': sku,
            'ABC_XYZ': abc_xyz[sku],
            'Current': current_stock[sku],
            'ROP': round(rop[sku]),
            'Order': round(qty),
            'Total': round(final_stock),
            'Cost': round(order_cost, 2)
        })

orders_df = pd.DataFrame(orders)

if len(orders_df) > 0:
    orders_df = orders_df.sort_values(by='Cost', ascending=False)

    print(f"\nТоваров к заказу: {len(orders_df)}")
    print(orders_df.to_string(index=False))

    total_order_cost = orders_df['Cost'].sum()
else:
    total_order_cost = 0

print(f"\nОбщая стоимость заказа: ${total_order_cost:,.2f}")



# 9. ИТОГОВЫЙ ОТЧЕТ

a_count = len(df_abc[df_abc['ABC'] == 'A'])
b_count = len(df_abc[df_abc['ABC'] == 'B'])
c_count = len(df_abc[df_abc['ABC'] == 'C'])

x_count = len(xyz_data[xyz_data['XYZ'] == 'X'])
y_count = len(xyz_data[xyz_data['XYZ'] == 'Y'])
z_count = len(xyz_data[xyz_data['XYZ'] == 'Z'])


print("ИТОГОВЫЙ ОТЧЕТ")


print(f"""
ДАТАСЕТ
Товаров всего: {len(df)} SKU
Общая выручка: ${total_revenue:,.0f}
Общие затраты: ${total_costs:,.0f}
Общая прибыль: ${total_profit:,.0f}

ABC АНАЛИЗ
Группа A: {a_count} товаров
Группа B: {b_count} товаров
Группа C: {c_count} товаров

XYZ АНАЛИЗ
X - стабильный спрос: {x_count} товаров
Y - умеренный спрос: {y_count} товаров
Z - нестабильный спрос: {z_count} товаров

КРИТИЧЕСКИЕ НАХОДКИ
Товаров в stockout: {len(stockout)}
Товаров ниже EOQ: {len(low_stock_eoq)} из {len(df)}
Товаров ниже ROP: {len(reorder_now)}
Критических товаров группы A ниже ROP: {len(critical_items)}

LP ОПТИМИЗАЦИЯ
Товаров требуют заказа: {len(orders_df)}
Стоимость пополнения: ${total_order_cost:,.2f}
""")
